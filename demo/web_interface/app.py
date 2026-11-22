from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# 必须是最早的 Streamlit 调用（下面有 @st.fragment 装饰器在导入期执行）
st.set_page_config(page_title="Campus Security Agent", layout="wide")

# 告警类型中文映射（后端 alert.type 形如 h03_detected / nl_rule_hit）
ALERT_TYPE_CN = {
    "NORMAL": "正常", "H01": "打架", "H02": "霸凌", "H03": "跌倒",
    "H04": "轻生", "H05": "破坏", "M01": "吸烟", "M02": "玩手机",
    "M03": "遮挡", "S01": "徘徊", "S02": "闯入",
    "NL_RULE_HIT": "规则命中",
}

# ===== 视觉主题色板（与下方全局 CSS 变量保持一致） =====
ACCENT = "#2563EB"       # 主强调色
COLOR_ALERT = "#DC2626"  # 告警/危险
COLOR_OK = "#16A34A"     # 确认/正常
COLOR_WARN = "#D97706"   # 警告/待定

# 视图模式：radio 选项（MODE_OPTIONS）与 ui_mode 取值一一对应，"idle" 为停止后的待机态
MODE_OPTIONS = ["native", "visualize", "zones"]
MODE_LABELS = {"native": "原生视频播放", "visualize": "可视化分析", "zones": "🎯 布控区域"}


def _on_ui_mode_change() -> None:
    """模式单选回调：同步 ui_mode；进入布控模式清空上次草图，进入可视化解除停止标记。"""
    picked = st.session_state.get("ui_mode_radio", "native")
    st.session_state["ui_mode"] = picked
    if picked == "visualize":
        st.session_state["task_stopped"] = False
    if picked == "zones":
        st.session_state.pop("zone_frame", None)
        st.session_state.pop("zone_points", None)


def humanize_alert_type(value: object) -> str:
    """h03_detected → 跌倒；nl_rule_hit → 规则命中；未知类型原样返回。"""
    raw = str(value or "").strip()
    if not raw:
        return "未知"
    key = raw.upper()
    for suffix in ("_DETECTED", "_ALERT"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return ALERT_TYPE_CN.get(key) or ALERT_TYPE_CN.get(raw.upper()) or raw


@st.fragment
def render_dashboard(api_base: str) -> None:
    """态势看板：按钮在片段内，点击只重跑本块，不重建左侧视频元素。"""
    c_dash_title, c_dash_btn = st.columns([4, 1])
    c_dash_title.subheader("📊 态势看板")
    c_dash_btn.button("⟳", key="dash_refresh", help="重新读取本地告警事件", use_container_width=True)
    try:
        import glob as _glob
        _event_files = sorted(_glob.glob(str(Path(__file__).resolve().parents[2] / "results" / "event_*.json")))
        _events = []
        for _ef in _event_files:
            try:
                _d = json.loads(Path(_ef).read_text(encoding="utf-8"))
                _a = _d.get("alert", {})
                if _a.get("triggered"):
                    _events.append({
                        "type": _a.get("type", "unknown").replace("_detected", "").upper(),
                        "decision": _a.get("final_decision", "pending"),
                        "ts": _d.get("timestamp", ""),
                        "frame_id": _d.get("frame_id", 0),
                        "source": _d.get("source", "pipeline"),
                    })
            except Exception:
                pass

        if _events:
            import pandas as _pd
            _df = _pd.DataFrame(_events)
            # 右列仅 1/3 宽：KPI 用 2×2 网格，避免四张卡片挤成一排
            _r1a, _r1b = st.columns(2)
            _r1a.metric("累计告警", len(_events))
            _r1b.metric("已确认", len(_df[_df["decision"] == "confirmed"]))
            _r2a, _r2b = st.columns(2)
            _r2a.metric("已驳回", len(_df[_df["decision"] == "rejected"]))
            _r2b.metric("待定", len(_df[_df["decision"] == "pending"]))
            st.caption("告警类型分布（按次数降序，横轴为次数）")
            _type_counts = _df["type"].map(humanize_alert_type).value_counts()
            st.altair_chart(horizontal_count_chart(_type_counts, COLOR_ALERT))
            st.caption("裁决状态占比（按次数降序，横轴为次数）")
            _dec_counts = _df["decision"].map(
                lambda value: DECISION_CN.get(str(value).lower(), str(value))
            ).value_counts()
            st.altair_chart(horizontal_count_chart(_dec_counts, COLOR_OK))
        else:
            st.info("暂无告警数据。运行分析任务后此处显示统计。")
    except Exception as exc:
        st.caption(f"看板读取失败: {exc}")


@st.fragment
def render_search(api_base: str) -> None:
    """场景检索：检索按钮在片段内，避免整页重跑导致视频重载。"""
    st.caption("本地知识库 · 自然语言检索")
    query_text = st.text_input("自然语言检索", value="", placeholder="例如: 有人在门口徘徊 / 跌倒")
    search_clicked = st.button("检索", use_container_width=True)
    if search_clicked and query_text.strip():
        try:
            search_resp = requests.get(
                f"{api_base}/api/v1/search",
                params={"q": query_text.strip(), "top_k": 10},
                timeout=30,
            )
            if search_resp.ok:
                st.session_state["search_result"] = search_resp.json()
            else:
                st.warning(f"检索接口失败: {search_resp.status_code}")
        except Exception as exc:
            st.warning(f"检索接口异常: {exc}")

    search_result = st.session_state.get("search_result")
    if isinstance(search_result, dict):
        st.caption(
            f"检索模式: {search_result.get('mode', '-')} | 索引文档数: {search_result.get('total_docs', 0)}"
        )
        hits = search_result.get("hits", [])
        if hits:
            for hit in hits:
                type_icon = "🧭" if hit.get("doc_type") == "scene_narration" else "🚨"
                time_label = hit.get("wall_time") or hit.get("video_time_sec") or "-"
                st.markdown(f"{type_icon} **[{hit.get('doc_type', '')}]** `{time_label}` (score={hit.get('score', 0)})")
                st.caption(str(hit.get("highlight", ""))[:200])
        else:
            st.info("没有命中结果。")


@st.fragment
def render_rules(api_base: str) -> None:
    """告警规则：刷新/增删都在片段内，点按钮不再触发整页重跑。"""
    c_rule_title, c_rule_btn = st.columns([4, 1])
    c_rule_title.subheader("📜 告警规则（自然语言定义）")
    refresh_rules_clicked = c_rule_btn.button("⟳", key="rules_refresh", help="刷新规则", use_container_width=True)
    new_rule_text = st.text_input(
        "一句话定义告警",
        value="",
        placeholder="例如: 告警有人翻越围栏 / 识别有人做出不文明手势",
    )
    add_rule_clicked = st.button("添加规则", use_container_width=True, type="primary")
    if add_rule_clicked:
        if new_rule_text.strip():
            with st.spinner("正在解析规则…"):
                t_parse0 = time.time()
                try:
                    created = requests.post(
                        f"{api_base}/api/v1/alert-rules",
                        json={"text": new_rule_text.strip()},
                        timeout=60,
                    )
                    parse_ms = round((time.time() - t_parse0) * 1000)
                    if created.ok:
                        rule_doc = created.json()
                        parsed = rule_doc.get("parsed", {})
                        if parsed.get("matched_classes"):
                            mode_label = "闭集映射 " + ",".join(parsed.get("matched_classes", []))
                        else:
                            mode_label = "开放词汇判定"
                        # 时刻①布控解析卡：解析结果 + 时延透明化（"改布控像发微信"的演示锚点）
                        st.success(f"✅ 规则已创建: {mode_label}（解析耗时 {parse_ms}ms）")
                        with st.expander("解析结果", expanded=True):
                            st.json({
                                "rule_id": rule_doc.get("rule_id", ""),
                                "matched_classes": parsed.get("matched_classes", []),
                                "open_description": parsed.get("open_description", ""),
                                "use_open_vocabulary": parsed.get("use_open_vocabulary", False),
                                "confidence_threshold": parsed.get("confidence_threshold"),
                                "解析耗时_ms": parse_ms,
                            })
                    else:
                        st.warning(f"创建失败: {created.text[:150]}")
                except Exception as exc:
                    st.warning(f"创建异常: {exc}")
        else:
            st.info("请输入规则内容。")

    def _reload_rules() -> None:
        st.session_state["alert_rules"] = requests.get(
            f"{api_base}/api/v1/alert-rules", timeout=15
        ).json()

    if refresh_rules_clicked:
        try:
            _reload_rules()
        except Exception as exc:
            st.warning(f"规则接口读取失败: {exc}")

    # 首次进入自动加载规则（只尝试一次，失败不反复打接口）
    if not isinstance(st.session_state.get("alert_rules"), dict) and not st.session_state.get("_rules_auto_tried"):
        st.session_state["_rules_auto_tried"] = True
        try:
            _reload_rules()
        except Exception as exc:
            st.caption(f"自动加载规则失败：{exc}，可点击右上角⟳重试。")

    rules_doc = st.session_state.get("alert_rules")
    if isinstance(rules_doc, dict):
        items = rules_doc.get("items", [])
        if items:
            for rule in items:
                rid = rule.get("rule_id", "")
                enabled = bool(rule.get("enabled", True))
                parsed = rule.get("parsed", {})
                matched_classes = parsed.get("matched_classes", [])
                if matched_classes:
                    mode = "闭集映射: " + ",".join(matched_classes)
                else:
                    mode = "开放词汇判定"
                head = "🟢" if enabled else "⚪"
                with st.container(border=True):
                    rule_text_col, rule_act_col = st.columns([5, 1])
                    rule_text_col.markdown(f"{head} **{rule.get('text', '')}**")
                    rule_text_col.caption(
                        f"`{rid}` ｜ {mode} ｜ 触发 {rule.get('stats', {}).get('triggered', 0)} 次"
                    )
                    if rule_act_col.button("停用" if enabled else "启用", key=f"rule_toggle_{rid}", use_container_width=True):
                        requests.put(
                            f"{api_base}/api/v1/alert-rules/{rid}",
                            json={"enabled": not enabled},
                            timeout=15,
                        )
                        st.session_state["alert_rules"] = requests.get(f"{api_base}/api/v1/alert-rules", timeout=15).json()
                        st.rerun(scope="fragment")
                    if rule_act_col.button("删除", key=f"rule_del_{rid}", use_container_width=True):
                        requests.delete(f"{api_base}/api/v1/alert-rules/{rid}", timeout=15)
                        st.session_state["alert_rules"] = requests.get(f"{api_base}/api/v1/alert-rules", timeout=15).json()
                        st.rerun(scope="fragment")
        else:
            st.info("暂无告警规则，输入一句话后点击“添加规则”。")
    else:
        st.info("点击“刷新规则”查看当前规则。")


@st.fragment
def render_logs(api_base: str) -> None:
    """识别日志 + 切片导出：按钮/表格编辑都在片段内。"""
    st.subheader("📋 识别日志（在线修改/导出）")
    c_log1, c_log2, c_log3 = st.columns([1, 1, 1])
    refresh_logs_clicked = c_log1.button("刷新历史告警")
    save_cache_clicked = c_log2.button("缓存日志到本地")
    load_cache_clicked = c_log3.button("加载本地缓存")

    def _reload_logs() -> None:
        alert_resp = requests.get(f"{api_base}/api/v1/alerts?limit=30", timeout=10)
        if alert_resp.ok:
            items = alert_resp.json().get("items", [])
            old_map = {row.get("log_id", ""): row for row in st.session_state["recognition_logs"]}
            merged_rows = []
            for item in items:
                row = event_to_log_row(item)
                old = old_map.get(row["log_id"])
                if old:
                    row["manual_label"] = old.get("manual_label", "")
                    row["manual_risk"] = old.get("manual_risk", "")
                    row["note"] = old.get("note", "")
                merged_rows.append(row)
            st.session_state["recognition_logs"] = merged_rows
        else:
            st.warning("获取告警列表失败")

    if refresh_logs_clicked:
        _reload_logs()

    # 首次进入自动加载历史告警（只尝试一次，失败不反复打接口）
    if not st.session_state.get("recognition_logs") and not st.session_state.get("_logs_auto_tried"):
        st.session_state["_logs_auto_tried"] = True
        try:
            _reload_logs()
        except Exception as exc:
            st.caption(f"自动加载历史告警失败：{exc}，可点击“刷新历史告警”重试。")

    cache_file = Path(__file__).resolve().parents[2] / "results" / "recognition_log_cache.json"
    if save_cache_clicked:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(st.session_state["recognition_logs"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        st.success(f"已缓存到: {cache_file}")

    if load_cache_clicked:
        if cache_file.exists():
            st.session_state["recognition_logs"] = json.loads(cache_file.read_text(encoding="utf-8"))
            st.success("已加载本地缓存日志")
        else:
            st.warning("本地缓存文件不存在")

    log_rows = st.session_state.get("recognition_logs", [])
    if log_rows:
        editable_df = pd.DataFrame(log_rows)
        # 列重排：常用信息在前、路径类在后；表头中文化
        log_column_order = [
            "frame_id", "timestamp", "alert_type", "risk_level", "final_decision", "triggered",
            "manual_label", "manual_risk", "note", "clip_path", "snapshot_url", "log_id",
        ]
        for _col in log_column_order:
            if _col not in editable_df.columns:
                editable_df[_col] = ""
        editable_df = editable_df[log_column_order]
        # frame_id 可能混有 "-" 占位，收敛为数值列（无法解析的置空），避免列类型冲突
        editable_df["frame_id"] = pd.to_numeric(editable_df["frame_id"], errors="coerce")
        edited_df = st.data_editor(
            editable_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="recognition_log_editor",
            column_config={
                "frame_id": st.column_config.NumberColumn("帧号", width="small"),
                "timestamp": st.column_config.TextColumn("时间戳", width="medium"),
                "alert_type": st.column_config.TextColumn("告警类型", width="small"),
                "risk_level": st.column_config.TextColumn("风险等级", width="small"),
                "final_decision": st.column_config.TextColumn("最终裁决", width="small"),
                "triggered": st.column_config.CheckboxColumn("已触发", disabled=True),
                "manual_label": st.column_config.TextColumn("人工标注", width="small"),
                "manual_risk": st.column_config.TextColumn("人工风险", width="small"),
                "note": st.column_config.TextColumn("备注", width="medium"),
                "clip_path": st.column_config.TextColumn("切片路径", width="medium"),
                "snapshot_url": st.column_config.TextColumn("快照", width="medium"),
                "log_id": st.column_config.TextColumn("日志ID", width="small"),
            },
            disabled=["log_id", "frame_id", "timestamp", "alert_type", "risk_level", "clip_path", "snapshot_url", "final_decision"],
        )
        st.session_state["recognition_logs"] = edited_df.to_dict(orient="records")

        json_bytes = json.dumps(st.session_state["recognition_logs"], ensure_ascii=False, indent=2).encode("utf-8")
        csv_bytes = edited_df.to_csv(index=False).encode("utf-8-sig")
        c_export1, c_export2 = st.columns([1, 1])
        c_export1.download_button(
            "导出识别日志 JSON",
            data=json_bytes,
            file_name="recognition_logs.json",
            mime="application/json",
            use_container_width=True,
        )
        c_export2.download_button(
            "导出识别日志 CSV",
            data=csv_bytes,
            file_name="recognition_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("暂无可编辑识别日志，先点击“刷新历史告警”或运行检测。")

    st.subheader("异常行为切片导出")
    abnormal_rows = [
        row for row in st.session_state.get("recognition_logs", []) if row.get("triggered") and row.get("clip_path")
    ]
    if abnormal_rows:
        export_rows = abnormal_rows[:10]
        # 两列网格排布，避免 10 个全宽按钮纵向堆叠
        for start in range(0, len(export_rows), 2):
            grid_cols = st.columns(2)
            for col, idx in zip(grid_cols, range(start, min(start + 2, len(export_rows)))):
                row = export_rows[idx]
                artifact_path = resolve_artifact_path(row.get("clip_path", ""))
                label = f"帧 {row.get('frame_id', '-')} | {humanize_alert_type(row.get('alert_type'))}"
                if artifact_path is None:
                    col.caption(f"{label} | 文件不存在")
                    continue
                col.download_button(
                    label=f"⬇ {label}",
                    data=artifact_path.read_bytes(),
                    file_name=artifact_path.name,
                    mime="video/mp4",
                    key=f"clip_export_{idx}_{row.get('log_id', idx)}",
                    use_container_width=True,
                )
    else:
        st.info("暂无可导出的异常行为切片记录。")


@st.fragment
def render_health(api_base: str, task_id: str | None) -> None:
    """系统健康/任务状态：卡片化 + 中文标签；原始响应收进折叠面板。"""
    c_title, c_refresh = st.columns([4, 1])
    c_title.subheader("🩺 系统健康 / 档位")
    refresh_clicked = c_refresh.button(
        "⟳", key="health_fragment_refresh", help="刷新健康状态", use_container_width=True)

    def _fetch_health() -> None:
        try:
            resp = requests.get(f"{api_base}/health", timeout=10)
            if resp.ok:
                st.session_state["health"] = resp.json()
            else:
                st.warning(f"读取 /health 失败: {resp.status_code}")
        except Exception as exc:
            st.warning(f"读取 /health 异常: {exc}")

    if refresh_clicked:
        _fetch_health()

    health = st.session_state.get("health")
    # 首次进入自动拉取一次，避免"必须先点按钮才有内容"
    if not isinstance(health, dict) and not st.session_state.get("_health_auto_tried"):
        st.session_state["_health_auto_tried"] = True
        _fetch_health()
        health = st.session_state.get("health")

    if isinstance(health, dict) and health:
        runtime = health.get("runtime") or {}
        inference = health.get("inference") or {}
        models = health.get("models") or {}
        preflight = health.get("preflight_summary") or {}

        status_raw = str(health.get("status", "unknown"))
        status_text = {"ok": "正常", "degraded": "降级"}.get(status_raw, status_raw)
        device = str(runtime.get("device", "-"))
        ep = str(runtime.get("execution_provider", "-")).replace("ExecutionProvider", "")
        ok_count = preflight.get("ok", "-")
        total = preflight.get("total", "-")
        missing = preflight.get("missing", 0)
        missing_int = int(missing) if isinstance(missing, (int, float)) else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("服务状态", status_text)
        k2.metric("推理档位", str(health.get("profile", "-")))
        k3.metric("计算设备", f"{device} · {ep}" if ep and ep != "-" else device)
        k4.metric("预检", f"{ok_count} / {total}",
                  delta="全部就绪" if missing_int == 0 else f"-{missing_int} 项",
                  delta_color="normal" if missing_int == 0 else "inverse")

        if missing_int > 0:
            detail = st.session_state.get("_preflight_detail")
            if not isinstance(detail, dict):
                try:
                    detail_resp = requests.get(f"{api_base}/api/v1/preflight", timeout=8)
                    detail = detail_resp.json() if detail_resp.ok else None
                    if isinstance(detail, dict):
                        st.session_state["_preflight_detail"] = detail
                except Exception:
                    detail = None
            if isinstance(detail, dict):
                missing_names = [str(c.get("name", "")) for c in detail.get("checks", [])
                                 if isinstance(c, dict) and not c.get("exists", True)]
                if missing_names:
                    st.caption("预检缺失: " + "、".join(missing_names) + "（缺失项会走回退逻辑）")

        st.markdown("**模型加载**")
        det_dbg = health.get("detector_debug") or {}
        pose_dbg = health.get("pose_debug") or {}
        act_dbg = health.get("action_debug") or {}
        model_rows = [
            ("检测", det_dbg, models.get("detector_onnx", "")),
            ("姿态", pose_dbg, models.get("pose_onnx", "")),
            ("动作", act_dbg, models.get("action_onnx", "")),
        ]
        for name, dbg, path in model_rows:
            loaded = bool(dbg.get("is_real_model_loaded"))
            fallback = bool(dbg.get("used_fallback"))
            error = str(dbg.get("error", "") or "")
            extra = ""
            if name == "动作":
                extra = "；PC 已加载" if dbg.get("pose_classifier_loaded") else "；PC 未加载"
                if dbg.get("slowfast_disabled"):
                    extra += "；SlowFast 已熔断"
            note = error[:80] if error else ("回退路径" if fallback else "正常")
            row1, row2, row3 = st.columns([1, 3, 4])
            row1.markdown(("✅ " if loaded else "⚠️ ") + f"**{name}**")
            row2.caption(Path(str(path)).name if path else "-")
            row3.caption(("⚠️ " if (fallback or error) else "") + note + extra)

        st.markdown("**运行参数**")
        st.caption(
            f"每 {inference.get('frame_stride', '-')} 帧处理 1 帧"
            f" · 姿态每 {inference.get('pose_stride', '-')} 帧"
            f" · 动作每 {inference.get('action_stride', '-')} 帧"
            f" · 帧率上限 {inference.get('max_fps', '-')}fps"
            f" · 姿态目标 ≤{inference.get('max_pose_tracks', '-')}"
            f" · 动作目标 ≤{inference.get('max_action_tracks', '-')}"
            f" · 动作窗口 {act_dbg.get('input_clip_len', '-')} 帧"
        )

        st.markdown("**实时自检**")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.caption(
                f"检测器 · 候选 {det_dbg.get('detections_before_threshold', '-')}"
                f" → 阈值后 {det_dbg.get('detections_after_threshold', '-')}"
                f" → NMS {det_dbg.get('detections_after_nms', '-')}"
                f"；有效人物 {det_dbg.get('effective_person_boxes', '-')}"
            )
        with d2:
            st.caption(
                f"姿态 · {pose_dbg.get('num_keypoints', '-')} 关键点"
                f"；{'真实模型' if pose_dbg.get('is_real_model_loaded') else '回退（合成关键点）'}"
            )
        with d3:
            st.caption(
                f"动作 · PC {'已加载' if act_dbg.get('pose_classifier_loaded') else '未加载'}"
                f"；SlowFast top1 {act_dbg.get('top1_label', '-')}({act_dbg.get('top1_score', '-')})"
                f"；缓冲 {act_dbg.get('buffer_len', '-')} 帧"
            )

        with st.expander("健康详情（原始响应）", expanded=False):
            st.json(health)
    else:
        st.info("尚未获取到健康状态：后端可能未启动，或 /health 接口不可达。")

    st.subheader("🎯 任务状态")
    if task_id:
        cached = st.session_state.get("_health_task_snapshot") or {}
        tried = cached.get("task_id") == task_id
        snapshot = cached.get("data") if tried else None
        # 每个任务只自动拉取一次（失败也不重复打接口），手动刷新可强制重取
        should_fetch = not tried
        if st.button("⟳ 刷新任务状态", key="health_task_refresh"):
            should_fetch = True
        if should_fetch:
            try:
                resp = requests.get(f"{api_base}/api/v1/tasks/{task_id}", timeout=10)
                if resp.ok:
                    st.session_state["_health_task_snapshot"] = {"task_id": task_id, "data": resp.json()}
                else:
                    st.session_state["_health_task_snapshot"] = {"task_id": task_id, "data": None}
                    st.warning(f"任务状态读取失败: {resp.status_code}")
            except Exception as exc:
                st.session_state["_health_task_snapshot"] = {"task_id": task_id, "data": None}
                st.warning(f"任务状态读取异常: {exc}")
            cached = st.session_state.get("_health_task_snapshot") or {}
            snapshot = cached.get("data") if cached.get("task_id") == task_id else None

        if isinstance(snapshot, dict) and snapshot:
            status = str(snapshot.get("status", "unknown"))
            status_icon = {"running": "🟢", "queued": "🟡", "finished": "✅", "failed": "❌", "stopped": "⏹️"}.get(status, "⚪")
            status_cn = {"running": "运行中", "queued": "排队中", "finished": "已完成",
                         "failed": "失败", "stopped": "已停止"}.get(status, status)
            st.caption(f"{status_icon} 任务 `{task_id}` · {status_cn}")
            stats = snapshot.get("stream_stats") or {}
            if isinstance(stats, dict) and stats:
                dropped_decode = int(stats.get("dropped_frames_decode", 0) or 0)
                dropped_result = int(stats.get("dropped_frames_result", 0) or 0)
                t1, t2, t3 = st.columns(3)
                t1.metric("已处理帧", stats.get("engine_process_frames", "-"))
                t2.metric("已推送帧", stats.get("pushed_frames", "-"))
                t3.metric("丢帧", dropped_decode + dropped_result)
                stages = stats.get("stage_ms") or {}
                if isinstance(stages, dict) and stages:
                    stage_cn = {"stage_a_decode": "解码", "stage_b_inference": "推理",
                                "stage_c_encode": "编码", "stage_d_push": "推送"}
                    parts = []
                    for key, val in stages.items():
                        if isinstance(val, dict):
                            parts.append(f"{stage_cn.get(key, key)} {round(float(val.get('avg_ms', 0.0)), 1)}ms")
                    if parts:
                        st.caption("阶段耗时 · " + " · ".join(parts))
                if dropped_decode or dropped_result:
                    st.caption(f"丢帧来源 · 解码 {dropped_decode} · 处理 {dropped_result}")
            if snapshot.get("error"):
                st.warning(f"任务错误: {snapshot.get('error')}")
            with st.expander("任务详情（原始响应）", expanded=False):
                st.json(snapshot)
        else:
            st.info(f"任务 `{task_id}` 暂无状态数据（后端可能已重启，任务记录已丢失）。")
    else:
        st.info("尚未启动任务")


def horizontal_count_chart(counts: "pd.Series", color: str = ACCENT):
    """计数分布 → 横向条形图。

    原来的 st.bar_chart 是竖向柱状图，类目挤在横轴上（NL_RULE_HIT / pending 这类
    中英混排长标签会被旋转、截断，读起来很费劲）。改成类目放纵轴、数值放横轴，
    中文标签横向排布不截断，并在柱尾标出计数，按数量从多到少排列。
    """
    import altair as alt

    frame = counts.rename_axis("label").reset_index(name="count")
    frame["count"] = frame["count"].astype(int)
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=alt.SortField(field="count", order="descending"), title=None,
                axis=alt.Axis(labelLimit=200, labelFontSize=13, labelPadding=6)),
        x=alt.X("count:Q", title="次数", axis=alt.Axis(tickMinStep=1, format="d", grid=True)),
    )
    bars = base.mark_bar(cornerRadius=3, color=color)
    labels = base.mark_text(align="left", dx=5, color="#475569", fontSize=12).encode(text="count:Q")
    return (bars + labels).properties(width="container", height=max(140, 34 * len(frame)))


# 告警裁决状态中文映射（态势看板 X 轴不再直接显示 pending/confirmed/rejected）
DECISION_CN = {"confirmed": "已确认", "rejected": "已驳回", "pending": "待定", "uncertain": "待复核"}

# 场景理解面板自动刷新周期（Streamlit fragment 的 run_every）
SCENE_REFRESH_INTERVAL = os.getenv("SCENE_REFRESH_SECONDS", "4s")
IMPORTANCE_LABEL = {"high": "🚨 重要", "normal": "常规"}


@st.fragment(run_every=SCENE_REFRESH_INTERVAL)
def render_scene_panel(api_base: str, task_id: str | None) -> None:
    """场景理解面板：自动轮询 + 重要理解主动推送（toast + 高亮）。

    - 只做增量提示：帧号大于"已提示帧"的重要条目才弹 toast，避免每轮刷新重复打扰
    - 接口失败时保留上次内容并降级为提示，不阻塞整页
    - 任务切换（帧号归零）时重置已提示帧，否则新任务前段的重要理解会被静音
    """
    st.subheader("🧠 场景理解（SceneNarrator）")

    current_task = st.session_state.get("task_id") or task_id or ""
    params: dict[str, object] = {"limit": 10}
    if current_task:
        params["task_id"] = current_task

    status: dict = st.session_state.get("scene_status") or {}
    items: list[dict] = st.session_state.get("scene_log") or []
    try:
        status = requests.get(f"{api_base}/api/v1/scene/status", timeout=8).json()
        items = requests.get(f"{api_base}/api/v1/scene/log", params=params, timeout=8).json().get("items", [])
        st.session_state["scene_status"] = status
        st.session_state["scene_log"] = items
    except Exception as exc:
        st.caption(f"⚠️ 场景理解接口暂不可用（{exc}），面板保持上次内容，恢复后自动更新。")

    narrator = status.get("narrator", {}) if isinstance(status, dict) else {}
    adjudicator = status.get("adjudicator", {}) if isinstance(status, dict) else {}
    if narrator:
        st.caption(
            f"🔄 自动刷新 {SCENE_REFRESH_INTERVAL} | 叙述器 "
            f"{'正常' if narrator.get('healthy') else '异常'}"
            f"（已叙述 {narrator.get('narration_count', 0)} 次 / 每 {narrator.get('interval_seconds', '-')} 秒一条）| "
            f"裁决器 {'正常' if adjudicator.get('healthy') else '异常'}"
        )
        if narrator.get("last_error"):
            st.caption(f"叙述器最近错误: {narrator.get('last_error')}")

    # 任务结束提示（低频检查，避免每 4s 打一次 tasks 接口）
    if current_task and time.time() - float(st.session_state.get("_scene_task_check_ts", 0.0)) > 20:
        st.session_state["_scene_task_check_ts"] = time.time()
        try:
            state = requests.get(f"{api_base}/api/v1/tasks/{current_task}", timeout=6).json().get("status", "")
            st.session_state["_scene_task_state"] = state
        except Exception:
            st.session_state["_scene_task_state"] = ""
    task_state = st.session_state.get("_scene_task_state", "")
    if task_state in ("finished", "failed", "stopped"):
        st.caption(f"ℹ️ 当前任务已{task_state}：场景叙述不再新增，运行新任务后自动延续。")

    if not items:
        st.info("暂无场景叙述：运行分析任务后每 20 秒视频时间自动生成一条，无需手动刷新。")
        return

    latest_frame = max(int(r.get("frame_id", 0) or 0) for r in items)
    if st.session_state.get("_scene_seen_task") != current_task:
        st.session_state["_scene_seen_task"] = current_task
        st.session_state["_scene_seen_frame"] = latest_frame
    seen_frame = int(st.session_state.get("_scene_seen_frame", 0) or 0)

    new_important = [
        r for r in items
        if str(r.get("importance", "")) == "high" and int(r.get("frame_id", 0) or 0) > seen_frame
    ]
    if new_important:
        newest = new_important[-1]
        st.toast(f"重要场景理解：{str(newest.get('scene_summary', ''))[:60]}", icon="🚨")
    st.session_state["_scene_seen_frame"] = max(seen_frame, latest_frame)

    latest = items[-1]
    importance = str(latest.get("importance", "normal"))
    st.markdown(f"**最新场景** {IMPORTANCE_LABEL.get(importance, importance)}")
    st.caption(
        f"第 {latest.get('frame_id', '-')} 帧 ｜ 视频 {latest.get('video_time_sec', '-')}s ｜ "
        f"{latest.get('wall_time', '')}"
    )
    if importance == "high":
        st.error(latest.get("scene_summary", ""))
        if latest.get("importance_reasons"):
            st.caption("推送依据: " + "、".join(str(x) for x in latest["importance_reasons"]))
    else:
        st.write(latest.get("scene_summary", ""))
    for activity in latest.get("notable_activities") or []:
        st.markdown(f"- ⚠️ {activity}")
    st.caption(
        f"环境: {latest.get('environment', '-')} | 在场人数: {latest.get('person_count', '-')} | "
        f"生成耗时: {latest.get('latency_ms', '-')}ms"
    )

    with st.expander("场景时间线（最近10条）"):
        for record in reversed(items):
            flag = "🚨 " if str(record.get("importance", "")) == "high" else ""
            st.markdown(
                f"- {flag}`{record.get('wall_time', '')}` "
                f"帧{record.get('frame_id', '-')}/视频{record.get('video_time_sec', '-')}s "
                f"{str(record.get('scene_summary', ''))[:120]}"
            )


st.title("Campus Security Agent")
st.caption("校园安全智能体 · 实时视频理解 · 自然语言告警布控")

# 全局视觉主题：统一字阶/卡片/色板；标题吸顶 + 视频列吸顶（滚动时保持可见）
st.markdown(
    """
    <style>
      /* ===== 主题变量（与 Python 侧色板常量保持一致） ===== */
      :root {
          --accent: #2563eb;
          --accent-soft: #eaf1fe;
          --danger: #dc2626;
          --ok: #16a34a;
          --text-main: #1f2937;
          --text-muted: #6b7280;
          --card-bg: #ffffff;
          --card-border: #e5e9f0;
          --card-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
          --radius: 12px;
      }
      /* ===== 字阶 ===== */
      h1 {
          font-size: 1.65rem;
          font-weight: 700;
          color: var(--text-main);
          position: sticky;
          top: 0;
          z-index: 100;
          background: rgba(255, 255, 255, 0.97);
          padding: 0.55rem 0.75rem;
          margin-bottom: 0.2rem;
          border-bottom: 1px solid var(--card-border);
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
      }
      h2 {
          font-size: 1.12rem;
          font-weight: 700;
          color: var(--text-main);
          border-left: 4px solid var(--accent);
          padding-left: 0.6rem;
      }
      /* 侧边栏标题不套主区的左侧强调条：改用细分隔线收束 */
      section[data-testid="stSidebar"] h2 {
          border-left: none;
          padding-left: 0;
          padding-bottom: 0.45rem;
          margin-bottom: 0.55rem;
          border-bottom: 1px solid var(--card-border);
          letter-spacing: 0.04em;
      }
      h3 {
          font-size: 1rem;
          font-weight: 600;
          color: var(--text-main);
      }
      [data-testid="stCaptionContainer"] { color: var(--text-muted); }
      /* ===== 卡片：KPI 指标 ===== */
      [data-testid="stMetric"] {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          padding: 0.6rem 0.9rem;
          box-shadow: var(--card-shadow);
      }
      [data-testid="stMetricValue"] { font-weight: 700; }
      [data-testid="stMetricLabel"] { color: var(--text-muted); }
      /* ===== 卡片：折叠面板 ===== */
      [data-testid="stExpander"] {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          box-shadow: var(--card-shadow);
      }
      /* ===== 分段式模式切换（视图模式 radio） ===== */
      div[role="radiogroup"] {
          background: #f3f5f9;
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          padding: 5px;
          gap: 6px;
      }
      div[role="radiogroup"] label {
          border-radius: 9px;
          padding: 6px 16px;
          font-weight: 500;
          transition: background 0.15s ease;
      }
      div[role="radiogroup"] label:hover { background: #e7ecf4; }
      div[role="radiogroup"] label:has(input:checked) {
          background: var(--accent-soft);
          box-shadow: inset 0 0 0 1px var(--accent);
      }
      div[role="radiogroup"] label:has(input:checked) span {
          color: var(--accent);
          font-weight: 600;
      }
      /* ===== 页签（管理类功能区） ===== */
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
          background: #f3f5f9;
          border: 1px solid var(--card-border);
          border-radius: var(--radius);
          padding: 5px;
          gap: 6px;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
          border-radius: 9px;
          padding: 8px 18px;
          font-weight: 500;
      }
      [data-testid="stTabs"] [aria-selected="true"] {
          background: var(--card-bg);
          color: var(--accent);
          box-shadow: var(--card-shadow);
      }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"],
      [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
      /* ===== 按钮 ===== */
      .stButton > button, .stDownloadButton > button {
          border-radius: 9px;
          font-weight: 500;
      }
      /* ===== 侧边栏 ===== */
      section[data-testid="stSidebar"] {
          background: #f8fafc;
          border-right: 1px solid var(--card-border);
      }
      /* 含可视化 iframe 的左列吸顶（可视化分析模式） */
      [data-testid="stColumn"]:has(iframe) {
          position: sticky;
          top: 64px;
          align-self: flex-start;
          z-index: 90;
          max-height: calc(100vh - 78px);
          overflow-y: auto;
          padding-right: 4px;
      }
      [data-testid="stColumn"]:has(iframe)::-webkit-scrollbar {
          width: 6px;
      }
      [data-testid="stColumn"]:has(iframe)::-webkit-scrollbar-thumb {
          background: #c9ced6;
          border-radius: 3px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_video_files(video_dir: str) -> list[str]:
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
    root = Path(video_dir)
    if not root.exists() or not root.is_dir():
        return []

    files = [str(path) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    files.sort()
    return files


def summarize_stream_source(value: object) -> str:
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return "-"
        normalized = trimmed.replace("\\", "/")
        return normalized.split("/")[-1] or trimmed
    return "-"


def build_ws_url(api_base: str, task_id: str) -> str:
    parsed = urlparse(api_base.strip())
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host}/ws/task/{task_id}"


def build_media_url(api_base: str, stream_url: str) -> str:
    if stream_url.startswith("http://") or stream_url.startswith("https://"):
        return stream_url

    stream_path = Path(stream_url)
    if stream_path.exists() and stream_path.is_file():
        encoded = quote(str(stream_path.resolve()), safe="")
        return f"{api_base.rstrip('/')}/api/v1/media?path={encoded}"

    return ""


def event_to_log_row(item: dict) -> dict:
    alert = item.get("alert", {}) if isinstance(item, dict) else {}
    frame_id = item.get("frame_id", "-")
    timestamp = item.get("timestamp", "")
    log_id = f"{frame_id}_{timestamp}"
    return {
        "log_id": log_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "alert_type": alert.get("type", "none"),
        "risk_level": "HIGH" if bool(alert.get("triggered", False)) else "LOW",
        "triggered": bool(alert.get("triggered", False)),
        "final_decision": alert.get("final_decision", "pending"),
        "clip_path": alert.get("clip_path", ""),
        "snapshot_url": alert.get("snapshot_url", ""),
        "manual_label": "",
        "manual_risk": "",
        "note": "",
    }


def resolve_artifact_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None

    base_dir = Path(__file__).resolve().parents[2]
    rel = raw_path.lstrip("/")
    candidates = [
        base_dir / rel,
        base_dir / "results" / rel,
        base_dir / "results" / Path(rel).name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def render_overlay_component(
    media_url: str,
    ws_url: str,
    align_offset_frames: int,
    debug_mode: bool,
    task_status: str = "unknown",
    task_error: str = "",
    stream_stats: dict | None = None,
    api_base: str = "",
) -> None:
    media_json = json.dumps(media_url)
    ws_json = json.dumps(ws_url)
    api_base_json = json.dumps(str(api_base or ""))
    align_offset_json = json.dumps(int(max(0, align_offset_frames)))
    debug_mode_json = json.dumps(bool(debug_mode))
    task_status_json = json.dumps(str(task_status or "unknown"))
    task_error_json = json.dumps(str(task_error or ""))
    stream_stats_json = json.dumps(stream_stats or {}, ensure_ascii=False)
    stats_specs_json = json.dumps(
        [
            {"key": "ws_state", "label": "连接状态", "tip": "前后端 WebSocket 连接状态"},
            {"key": "task_status", "label": "任务状态", "tip": "后端任务当前状态"},
            {"key": "task_error", "label": "任务错误", "tip": "后端任务失败原因（如有）"},
            {"key": "event_count", "label": "事件总数", "tip": "前端收到的后端事件总数"},
            {"key": "draw_count", "label": "绘制帧数", "tip": "前端已经渲染的叠层帧数"},
            {"key": "video_paused", "label": "视频暂停", "tip": "浏览器原生视频当前是否暂停"},
            {"key": "video_time_s", "label": "播放时间(s)", "tip": "当前播放时间，单位秒"},
            {"key": "source_fps", "label": "源帧率", "tip": "后端捕获到的视频源帧率"},
            {"key": "max_fps", "label": "上限帧率", "tip": "任务配置中的最大输出帧率"},
            {"key": "frame_stride", "label": "帧跳步长", "tip": "后端推理的帧跳步长"},
            {"key": "push_stride", "label": "推送步长", "tip": "后端向前端推送事件的步长"},
            {"key": "read_frames", "label": "读取帧数", "tip": "后端已经从视频源读取的帧数"},
            {"key": "engine_process_frames", "label": "实际推理帧数", "tip": "真正调用模型完成推理的帧数"},
            {"key": "replayed_frames", "label": "复用帧数", "tip": "复用上一条推理结果直接输出的帧数"},
            {"key": "inference_frames", "label": "处理帧数", "tip": "进入推理循环处理的总帧数，包含复用帧"},
            {"key": "encoded_frames", "label": "编码帧数", "tip": "已经完成可视化编码的帧数"},
            {"key": "pushed_frames", "label": "推送帧数", "tip": "已经推送到前端的帧数"},
            {"key": "dropped_decode", "label": "解码丢帧", "tip": "读取队列中被挤掉的帧数"},
            {"key": "dropped_result", "label": "结果丢帧", "tip": "结果队列中被挤掉的帧数"},
            {"key": "dropped_packet", "label": "发送丢帧", "tip": "发送队列中被挤掉的帧数"},
            {"key": "event_frame_id", "label": "事件帧号", "tip": "当前用于叠层显示的事件帧号"},
            {"key": "detections", "label": "检测数", "tip": "当前帧中识别到的目标数量"},
            {"key": "pipeline_compute_ms", "label": "推理耗时(ms)", "tip": "后端单帧推理的总耗时"},
            {"key": "stage_detector_ms", "label": "检测耗时(ms)", "tip": "目标检测阶段耗时"},
            {"key": "stage_pose_ms", "label": "姿态耗时(ms)", "tip": "姿态估计阶段耗时"},
            {"key": "stage_action_ms", "label": "行为耗时(ms)", "tip": "行为识别阶段耗时"},
            {"key": "stage_post_ms", "label": "后处理耗时(ms)", "tip": "后处理与告警阶段耗时"},
            {"key": "stage_sum_ms", "label": "阶段总和(ms)", "tip": "各阶段耗时求和"},
        ],
        ensure_ascii=False,
    )
    stats_panel_markup = ""
    if debug_mode:
        stats_panel_markup = """
    <div id=\"statsPanel\" style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px;padding:10px 12px 16px;background:#f2f2f2;color:#1f2937;border-top:1px solid #d8d8d8;font:12px/1.45 'Consolas','Courier New',monospace;box-sizing:border-box;\"></div>
"""

    html = f"""
<div style=\"border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#111;\" id=\"outerBox\">
  <div id=\"wrap\" style=\"position:relative;width:100%;background:#000;\">
    <video id=\"nativeVideo\" src=\"\" controls autoplay muted playsinline style=\"width:100%;height:auto;display:block;\"></video>
    <canvas id=\"overlayCanvas\" style=\"position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;\"></canvas>
    <div id=\"diagBanner\" style=\"display:none;position:absolute;top:34px;left:10px;right:10px;z-index:9;\"></div>
    <button id=\"floatBtn\" style=\"position:absolute;top:8px;right:8px;z-index:10;display:none;background:rgba(31,78,121,0.92);color:#fff;border:none;border-radius:6px;padding:5px 10px;font:12px/1.4 sans-serif;cursor:pointer;\">🧷 悬浮窗口</button>
  </div>
{stats_panel_markup}
</div>
<script>
(() => {{
  const _v = '3.1';
  const _dbg = document.getElementById('diagBanner');
  const _showErr = (msg) => {{
    try {{
      if (_dbg) {{
        _dbg.style.display = 'block';
        _dbg.innerHTML = msg;
      }}
    }} catch (e) {{}}
  }};
  _showErr('<div style="background:rgba(31,78,121,0.85);color:#fff;border-radius:6px;padding:4px 10px;font:11px sans-serif;">overlay v' + _v + ' loading...</div>');
  try {{
  const mediaUrl = {media_json};
  const wsUrl = {ws_json};
  const apiBase = {api_base_json};
  // 检测框标签中文化：行为编号 → 中文名，角色 → 中文名
  const LABEL_CN = {{Normal:'正常', H01:'打架', H02:'霸凌', H03:'跌倒', H04:'轻生', H05:'破坏', M01:'吸烟', M02:'玩手机', M03:'遮挡', S01:'徘徊', S02:'闯入'}};
  const ROLE_CN = {{attacker:'攻击者', victim:'受害者', bystander:'旁观者'}};
    const alignOffsetFrames = {align_offset_json};
    const debugMode = {debug_mode_json};
        const initialTaskStatus = {task_status_json};
        const initialTaskError = {task_error_json};
        const initialStreamStats = {stream_stats_json};
    const statsSpecs = {stats_specs_json};
  const video = document.getElementById('nativeVideo');
  const canvas = document.getElementById('overlayCanvas');
    const statsPanel = document.getElementById('statsPanel');
    const statsNodes = {{}};
  const ctx = canvas.getContext('2d');
  // ── 播放位置跨重跑保持：Streamlit 重跑会重建组件，父页面 window 不刷新 ──
  const RESUME_KEY = '__huiyanVideoResumeTime';
  let resumeAt = 0;
  let resumeApplied = false;
  let pendingResumeTime = null;
  try {{ resumeAt = Number(window.parent[RESUME_KEY]) || 0; }} catch (e) {{ resumeAt = 0; }}
  function capturePlayhead() {{
      pendingResumeTime = (typeof video.currentTime === 'number' && video.currentTime > 0.5) ? video.currentTime : null;
      resumeApplied = false;  // 跨文档搬运触发新的媒体加载，允许重新应用恢复位置
  }}
  video.src = mediaUrl;
    video.currentTime = 0;
    video.addEventListener('loadedmetadata', () => {{
                resizeCanvas();
        const target = (pendingResumeTime !== null && pendingResumeTime > 0.5) ? pendingResumeTime : resumeAt;
        if (target > 0.5 && !resumeApplied) {{
            resumeApplied = true;
            try {{ video.currentTime = target; }} catch (e) {{}}
            pendingResumeTime = null;
        }}
        autoPlay();
    }});
    // 播放中持续把进度写回父页面（供下次组件重建时恢复）
    video.addEventListener('timeupdate', () => {{
        try {{ window.parent[RESUME_KEY] = video.currentTime; }} catch (e) {{}}
    }});

        if (debugMode && statsPanel) {{
                statsPanel.innerHTML = statsSpecs.map((spec) => `
                        <div class=\"stat-row\" style=\"display:grid;grid-template-columns:108px minmax(0,1fr);gap:10px;align-items:center;min-height:26px;padding:6px 8px;background:#fafafa;border:1px solid #e3e3e3;border-radius:6px;box-sizing:border-box;\" title=\"${{spec.tip}}\">
                            <span class=\"stat-label\" style=\"color:#4b5563;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;\">${{spec.label}}</span>
                            <span class=\"stat-value\" id=\"stat-${{spec.key}}\" style=\"color:#111827;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums;text-align:right;\">-</span>
                        </div>
                `).join('');
                for (const spec of statsSpecs) {{
                        statsNodes[spec.key] = document.getElementById(`stat-${{spec.key}}`);
                }}
        }}

        let eventBuffer = [];
        let latestEvent = null;
        let sourceFps = null;
    let firstFrameReady = false;
        let userPaused = false;
        let suppressPauseEvent = false;
        let suppressPlayEvent = false;
        let syncEnabled = true;
        let waitingForSync = true;
        let backendTaskStatus = initialTaskStatus || 'unknown';
        let backendTaskError = initialTaskError || '';
        let backendStreamStats = initialStreamStats || {{}};
        let lastRenderedKey = '';
        const syncLeadFrames = 0;
        let lastEventAtMs = performance.now();
        const noEventReleaseMs = 1200;
        const staleHoldFrames = 2;
        let wsState = 'connecting';
  let eventCount = 0;
  let drawCount = 0;
  let lastStatsMs = 0;
    let mediaError = false;
    let apiProbe = 'unknown';
    let reconnectTimer = null;
    let wsRetry = 0;
    const diagBanner = document.getElementById('diagBanner');
    const floatBtn = document.getElementById('floatBtn');

    // 无 CORS 模式探测 API 可达性：网络失败 reject（不可达），成功返回 opaque 响应（可达）
    fetch(apiBase + '/health', {{ mode: 'no-cors' }})
        .then(() => {{ apiProbe = 'reachable'; }})
        .catch(() => {{ apiProbe = 'unreachable'; }});
    video.addEventListener('error', () => {{ mediaError = true; }});

    function pushEvent(evt) {{
        if (!evt || typeof evt.frame_id !== 'number') return;
        if (eventBuffer.length > 0 && evt.frame_id <= eventBuffer[eventBuffer.length - 1].frame_id) {{
            return;
        }}

        eventBuffer.push(evt);
        if (eventBuffer.length > 5000) {{
            eventBuffer = eventBuffer.slice(eventBuffer.length - 5000);
        }}
    }}

    function getTimelineFps() {{
        return (typeof sourceFps === 'number' && sourceFps > 1) ? sourceFps : 15.0;
    }}

    function selectEventByVideoTime() {{
        if (eventBuffer.length === 0) return null;
        const timelineFps = getTimelineFps();
        const targetFrameId = Math.max(1, Math.floor(video.currentTime * timelineFps) + 1 + alignOffsetFrames);
        const newestFrameId = eventBuffer[eventBuffer.length - 1].frame_id;
        if (targetFrameId > newestFrameId + staleHoldFrames) {{
            return null;
        }}

        let selected = eventBuffer[0];
        for (let i = eventBuffer.length - 1; i >= 0; i -= 1) {{
            if (eventBuffer[i].frame_id <= targetFrameId) {{
                selected = eventBuffer[i];
                break;
            }}
        }}
        return selected;
    }}

    // 相邻两帧结果对：a=当前时间之前的最近结果，b=之后的结果，t=插值比例。
    // 推理只有 ~2fps，若只取 a，框会在两次结果之间完全静止（人走开框不动）。
    function selectEventPairByVideoTime() {{
        if (eventBuffer.length === 0) return {{ a: null, b: null, t: 0 }};
        const timelineFps = getTimelineFps();
        const targetFrameId = Math.max(1, Math.floor(video.currentTime * timelineFps) + 1 + alignOffsetFrames);
        const newestFrameId = eventBuffer[eventBuffer.length - 1].frame_id;
        if (targetFrameId > newestFrameId + staleHoldFrames) {{
            return {{ a: null, b: null, t: 0 }};
        }}
        let ai = -1;
        for (let i = eventBuffer.length - 1; i >= 0; i -= 1) {{
            if (eventBuffer[i].frame_id <= targetFrameId) {{ ai = i; break; }}
        }}
        if (ai < 0) return {{ a: eventBuffer[0], b: eventBuffer[0], t: 0 }};
        const a = eventBuffer[ai];
        const b = (ai + 1 < eventBuffer.length) ? eventBuffer[ai + 1] : a;
        let t = 0;
        if (b !== a && Number(b.frame_id) > Number(a.frame_id)) {{
            t = Math.min(1, Math.max(0,
                (targetFrameId - Number(a.frame_id)) / (Number(b.frame_id) - Number(a.frame_id))));
        }}
        return {{ a: a, b: b, t: t }};
    }}

    function updateStat(key, value) {{
        const node = statsNodes[key];
        if (node) {{
            node.textContent = value;
        }}
    }}

    function setProgrammaticPause() {{
        if (!video.paused) {{
            suppressPauseEvent = true;
            video.pause();
        }}
    }}

    function autoPlay() {{
        if (video.paused) {{
            suppressPlayEvent = true;
            video.play().catch(() => {{}});
        }}
    }}

    const maxLagFrames = 6;
    function maybeSyncPlayback() {{
        if (!firstFrameReady || eventBuffer.length === 0) return;
        if (userPaused) return;

        const nowMs = performance.now();
        const timelineFps = getTimelineFps();
        const targetFrameId = Math.max(1, Math.floor(video.currentTime * timelineFps) + 1);
        const newestFrameId = eventBuffer[eventBuffer.length - 1].frame_id;
        const lagFrames = targetFrameId - newestFrameId;

        // 滞后保护：同步被用户播放/暂停事件关闭后，如果框已落后超过 6 帧，
        // 自动重新接管（否则 VLM 占用 GPU 时视频会一路跑到结果前面，框跟不上人）。
        if (!syncEnabled) {{
            if (lagFrames <= maxLagFrames) return;
            syncEnabled = true;
        }}

        if ((nowMs - lastEventAtMs) > noEventReleaseMs) {{
            waitingForSync = false;
            if (video.paused) {{
                autoPlay();
            }}
            return;
        }}

        if (wsState === 'closed' || wsState === 'error') {{
            waitingForSync = false;
            if (video.paused) {{
                autoPlay();
            }}
            return;
        }}

        const caughtUp = targetFrameId <= (newestFrameId - syncLeadFrames);

        if (!caughtUp) {{
            waitingForSync = true;
            setProgrammaticPause();
            return;
        }}

        if (waitingForSync && video.paused) {{
            waitingForSync = false;
            autoPlay();
        }}
    }}

  function resizeCanvas() {{
    const w = Math.max(1, Math.floor(video.clientWidth));
    const h = Math.max(1, Math.floor(video.clientHeight));
    if (canvas.width !== w || canvas.height !== h) {{
      canvas.width = w;
      canvas.height = h;
    }}
  }}

  function drawOverlay() {{
    maybeSyncPlayback();
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

        const eventPair = selectEventPairByVideoTime();
        const syncedEvent = eventPair.a;
        const nextEvent = eventPair.b;
        const interpT = eventPair.t;
        latestEvent = syncedEvent;
        const frameIdForRender = syncedEvent && syncedEvent.frame_id ? syncedEvent.frame_id : '-';
        const renderKey = `${{video.currentTime.toFixed(3)}}|${{frameIdForRender}}|${{video.paused}}`;
        if (renderKey !== lastRenderedKey) {{
            drawCount += 1;
            lastRenderedKey = renderKey;
        }}

        if (syncedEvent && Array.isArray(syncedEvent.detections)) {{
            const dets = syncedEvent.detections;
            const streaming = syncedEvent.metrics && syncedEvent.metrics.streaming ? syncedEvent.metrics.streaming : {{}};
      const inputSize = Array.isArray(streaming.input_size) ? streaming.input_size : [];
      const refW = inputSize.length >= 2 ? Math.max(1, Number(inputSize[0])) : Math.max(1, video.videoWidth || canvas.width);
      const refH = inputSize.length >= 2 ? Math.max(1, Number(inputSize[1])) : Math.max(1, video.videoHeight || canvas.height);
      const sx = canvas.width / refW;
      const sy = canvas.height / refH;

      for (const det of dets) {{
                const rawBbox = Array.isArray(det.bbox) ? det.bbox : [];
        if (rawBbox.length !== 4) continue;
        let bx = Number(rawBbox[0]), by = Number(rawBbox[1]), bw = Number(rawBbox[2]), bh = Number(rawBbox[3]);
        // 向下一帧结果线性插值：把 ~2fps 的离散框补成连续运动
        if (nextEvent && nextEvent !== syncedEvent && interpT > 0.01 && Array.isArray(nextEvent.detections)) {{
            const nd = nextEvent.detections.find(d => String(d.track_id) === String(det.track_id));
            if (nd && Array.isArray(nd.bbox) && nd.bbox.length === 4) {{
                bx += (Number(nd.bbox[0]) - bx) * interpT;
                by += (Number(nd.bbox[1]) - by) * interpT;
                bw += (Number(nd.bbox[2]) - bw) * interpT;
                bh += (Number(nd.bbox[3]) - bh) * interpT;
            }}
        }}
        const x1 = Math.max(0, bx * sx);
        const y1 = Math.max(0, by * sy);
        const x2 = Math.min(canvas.width - 1, (bx + bw) * sx);
        const y2 = Math.min(canvas.height - 1, (by + bh) * sy);

        const behavior = det.behavior || {{}};
        const risk = behavior.risk_level || 'LOW';
        let color = '#00c853';
        if (risk === 'HIGH') color = '#ff1744';
        else if (risk === 'MEDIUM') color = '#ff9100';

        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));

        const actionName = LABEL_CN[behavior.type] || behavior.type || '正常';
        const roleName = (behavior.role && behavior.role !== 'unknown') ? (ROLE_CN[behavior.role] || behavior.role) : '';
        const label = `${{det.track_id || '-'}} ${{actionName}}${{roleName ? '·' + roleName : ''}} ${{Number(behavior.confidence || 0).toFixed(2)}}`;
        ctx.font = '13px sans-serif';
        const textW = ctx.measureText(label).width + 10;
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - 18), textW, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x1 + 5, Math.max(12, y1 - 5));
      }}

            const alert = syncedEvent.alert || {{}};
      if (alert.triggered) {{
        ctx.fillStyle = '#d50000';
        ctx.fillRect(8, 8, 340, 28);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px sans-serif';
        const rawAlertType = String(alert.type || '');
        const alertKey = rawAlertType.toUpperCase().replace(/_DETECTED$/, '').replace(/_ALERT$/, '');
        const alertName = LABEL_CN[alertKey] || (alertKey === 'NL_RULE_HIT' ? '规则命中' : (rawAlertType || '未知'));
        ctx.fillText(`ALERT: ${{alertName}}`, 14, 27);
      }}
    }}

    const now = performance.now();
    if (debugMode && statsPanel && now - lastStatsMs > 250) {{
            const detCount = latestEvent && Array.isArray(latestEvent.detections) ? latestEvent.detections.length : 0;
            const frameId = latestEvent && latestEvent.frame_id ? latestEvent.frame_id : '-';
            const latency = latestEvent && latestEvent.metrics ? latestEvent.metrics.pipeline_latency_ms : '-';
            const runtime = latestEvent && latestEvent.metrics ? (latestEvent.metrics.stream_runtime || backendStreamStats || {{}}) : (backendStreamStats || {{}});
            const stage = latestEvent && latestEvent.metrics ? (latestEvent.metrics.pipeline_stage_ms || {{}}) : {{}};
            const streaming = latestEvent && latestEvent.metrics ? (latestEvent.metrics.streaming || {{}}) : {{}};
            const pushed = (typeof runtime.pushed_frames === 'number') ? runtime.pushed_frames : '-';
            const read = (typeof runtime.read_frames === 'number') ? runtime.read_frames : '-';
            updateStat('ws_state', wsState);
            updateStat('task_status', backendTaskStatus);
            updateStat('task_error', backendTaskError || '-');
            updateStat('event_count', String(eventCount));
            updateStat('draw_count', String(drawCount));
            updateStat('video_paused', String(video.paused));
            updateStat('video_time_s', video.currentTime.toFixed(3));
            updateStat('source_fps', sourceFps !== null ? sourceFps.toFixed(3) : '-');
            updateStat('max_fps', streaming.max_fps !== undefined ? String(streaming.max_fps) : '-');
            updateStat('frame_stride', streaming.frame_stride !== undefined ? String(streaming.frame_stride) : '-');
            updateStat('push_stride', streaming.push_stride !== undefined ? String(streaming.push_stride) : '-');
            updateStat('read_frames', String(read));
            updateStat('engine_process_frames', runtime.engine_process_frames !== undefined ? String(runtime.engine_process_frames) : '-');
            updateStat('replayed_frames', runtime.replayed_frames !== undefined ? String(runtime.replayed_frames) : '-');
            updateStat('inference_frames', runtime.inference_frames !== undefined ? String(runtime.inference_frames) : '-');
            updateStat('encoded_frames', runtime.encoded_frames !== undefined ? String(runtime.encoded_frames) : '-');
            updateStat('pushed_frames', String(pushed));
            updateStat('dropped_decode', runtime.dropped_frames_decode !== undefined ? String(runtime.dropped_frames_decode) : '-');
            updateStat('dropped_result', runtime.dropped_frames_result !== undefined ? String(runtime.dropped_frames_result) : '-');
            updateStat('dropped_packet', runtime.dropped_frames_packet !== undefined ? String(runtime.dropped_frames_packet) : '-');
            updateStat('event_frame_id', String(frameId));
            updateStat('detections', String(detCount));
            updateStat('pipeline_compute_ms', String(latency));
            updateStat('stage_detector_ms', stage.detector ?? '-');
            updateStat('stage_pose_ms', stage.pose ?? '-');
            updateStat('stage_action_ms', stage.action ?? '-');
            updateStat('stage_post_ms', stage.post ?? '-');
            updateStat('stage_sum_ms', stage.sum_stages ?? '-');
      lastStatsMs = now;
    }}

        updateDiagBanner();

    function updateDiagBanner() {{
        if (!diagBanner) return;
        const msgs = [];
        if (mediaError) {{
            msgs.push(['⚠️ 视频加载失败：本地浏览器无法访问媒体地址（' + apiBase + '）',
                       '请确认 SSH 隧道已转发 8000 端口，例如: ssh -N -L 8501:localhost:8501 -L 8000:localhost:8000 user@server']);
        }} else if (wsState !== 'open' && performance.now() - lastEventAtMs > 2500) {{
            let t = '⚠️ API/WebSocket 未连接（' + wsState + '）→ ' + apiBase;
            if (apiProbe === 'unreachable') t += ' —— 探测失败：请确认 SSH 隧道已转发 8000 端口';
            msgs.push([t, '连接恢复后事件回放自动继续']);
        }}
        if (msgs.length === 0) {{ diagBanner.style.display = 'none'; return; }}
        let html = '';
        for (const m of msgs) {{
            html += '<div style="background:rgba(202,21,21,0.94);color:#fff;border-radius:8px;padding:8px 12px;margin-bottom:6px;box-shadow:0 3px 10px rgba(0,0,0,0.3);">'
                 +  '<div style="font-weight:bold;">' + m[0] + '</div>'
                 +  '<div style="font-size:12px;opacity:.9;">' + m[1] + '</div></div>';
        }}
        diagBanner.innerHTML = html;
        diagBanner.style.display = 'block';
    }}

    requestAnimationFrame(drawOverlay);
  }}

  function onWsMessage(ev) {{
    try {{
            const parsed = JSON.parse(ev.data);
            const streaming = parsed && parsed.metrics && parsed.metrics.streaming ? parsed.metrics.streaming : {{}};
            if (streaming && Number(streaming.source_fps) > 1) {{
                sourceFps = Number(streaming.source_fps);
            }}
            pushEvent(parsed);
      eventCount += 1;
            lastEventAtMs = performance.now();
            maybeSyncPlayback();

            if (!firstFrameReady && parsed && parsed.frame_id === 1) {{
                firstFrameReady = true;
                userPaused = false;
                waitingForSync = false;
                if (resumeAt <= 0.5 && !resumeApplied) {{
                    video.currentTime = 0;
                }}
                autoPlay();
            }} else if (!firstFrameReady && eventBuffer.length > 0) {{
                firstFrameReady = true;
                userPaused = false;
                waitingForSync = false;
                if (resumeAt <= 0.5 && !resumeApplied) {{
                    video.currentTime = 0;
                }}
                autoPlay();
            }}
    }} catch (e) {{
      wsState = 'parse_error';
    }}
  }}

  // ── 带自动重连的 WS 连接（指数退避，恢复后服务端自动回放事件） ──
  let ws = null;
  function connectWs() {{
    wsState = 'connecting';
    try {{
      ws = new WebSocket(wsUrl);
    }} catch (e) {{
      wsState = 'error';
      scheduleReconnect();
      return;
    }}
    ws.onopen = () => {{
      wsState = 'open';
      wsRetry = 0;
      ws.send('start');
    }};
    ws.onmessage = onWsMessage;
    ws.onerror = () => {{ wsState = 'error'; }};
    ws.onclose = () => {{
      wsState = 'closed';
      scheduleReconnect();
    }};
  }}
  function scheduleReconnect() {{
    if (reconnectTimer) return;
    const delay = Math.min(8000, 800 * Math.pow(2, wsRetry++));
    reconnectTimer = setTimeout(() => {{
      reconnectTimer = null;
      connectWs();
    }}, delay);
  }}
  connectWs();

    video.addEventListener('pause', () => {{
        if (suppressPauseEvent) {{
            suppressPauseEvent = false;
            return;
        }}
        userPaused = true;
        waitingForSync = false;
        syncEnabled = false;   // 用户手动暂停：自动同步从此不再干预
    }});
    video.addEventListener('play', () => {{
        if (suppressPlayEvent) {{
            suppressPlayEvent = false;
            return;
        }}
        userPaused = false;
        syncEnabled = false;   // 用户手动播放：自动同步从此不再干预
    }});

  window.addEventListener('resize', resizeCanvas);
    video.addEventListener('seeking', () => {{ maybeSyncPlayback(); }});
  video.addEventListener('loadedmetadata', resizeCanvas);
  requestAnimationFrame(drawOverlay);

  // ── 悬浮窗口：把视频画面挂到页面视口固定位置，滚动时保持可见 ──
  const wrapEl = document.getElementById('wrap');
  let floating = false;
  let floatHost = null;
  const dragState = {{ dragging: false, offsetX: 0, offsetY: 0 }};
  function toggleFloat() {{
    let parentDoc = null;
    try {{ parentDoc = window.parent.document; }} catch (e) {{ parentDoc = null; }}
    if (!parentDoc) return;
    if (!floating) {{
      capturePlayhead();
      floatHost = parentDoc.createElement('div');
      floatHost.style.cssText = 'position:fixed;top:74px;right:20px;width:min(430px,42vw);z-index:999900;border:2px solid #1F4E79;border-radius:10px;overflow:hidden;box-shadow:0 8px 28px rgba(0,0,0,0.35);background:#000;';
      const bar = parentDoc.createElement('div');
      bar.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:4px 10px;background:#1F4E79;color:#fff;font:12px/1.7 sans-serif;';
      const label = parentDoc.createElement('span');
      label.textContent = '🧷 悬浮监控画面（滚动时保持可见）';
      const restoreBtn = parentDoc.createElement('button');
      restoreBtn.textContent = '还原到页面';
      restoreBtn.style.cssText = 'border:none;background:rgba(255,255,255,0.22);color:#fff;border-radius:5px;padding:2px 9px;cursor:pointer;font:12px sans-serif;';
      restoreBtn.addEventListener('click', toggleFloat);
      bar.appendChild(label);
      bar.appendChild(restoreBtn);
      floatHost.appendChild(bar);
      parentDoc.body.appendChild(floatHost);
      // ── 标题栏拖拽：按住标题栏拖动悬浮窗口到任意位置 ──
      bar.style.cursor = 'move';
      bar.addEventListener('pointerdown', (e) => {{
          if (e.target === restoreBtn) return;
          const rect = floatHost.getBoundingClientRect();
          dragState.offsetX = e.clientX - rect.left;
          dragState.offsetY = e.clientY - rect.top;
          dragState.dragging = true;
          // 切换到 left 定位（初始用 right，拖拽时需转为 left）
          floatHost.style.right = 'auto';
          floatHost.style.left = rect.left + 'px';
          try {{ bar.setPointerCapture(e.pointerId); }} catch (e2) {{}}
          e.preventDefault();
      }});
      bar.addEventListener('pointermove', (e) => {{
          if (!dragState.dragging || !floatHost) return;
          const maxX = parentDoc.body.clientWidth - 100;
          const x = Math.max(0, Math.min(maxX, e.clientX - dragState.offsetX));
          const y = Math.max(0, e.clientY - dragState.offsetY);
          floatHost.style.left = x + 'px';
          floatHost.style.top = y + 'px';
      }});
      bar.addEventListener('pointerup', () => {{ dragState.dragging = false; }});
      floatHost.appendChild(parentDoc.adoptNode(wrapEl));
      // 跨文档搬运会让视频重新加载：恢复播放位置并继续播放
      try {{ autoPlay(); }} catch (e) {{}}
      floating = true;
      if (floatBtn) floatBtn.textContent = '⤺ 还原到页面';
    }} else {{
      capturePlayhead();
      try {{
        const outerBox = document.getElementById('outerBox');
        outerBox.insertBefore(parentDoc.adoptNode(wrapEl), outerBox.firstChild);
      }} catch (e) {{}}
      if (floatHost) {{ floatHost.remove(); floatHost = null; }}
      try {{ autoPlay(); }} catch (e) {{}}
      floating = false;
      if (floatBtn) floatBtn.textContent = '🧷 悬浮窗口';
    }}
  }}
  if (floatBtn) {{
    try {{
      window.parent.document; // 同源探测：跨源时隐藏按钮
      floatBtn.style.display = 'block';
      floatBtn.addEventListener('click', toggleFloat);
    }} catch (e) {{
      floatBtn.style.display = 'none';
    }}
  }}
  // 组件被 Streamlit 重载/移除时，清理遗留的悬浮窗口
  const cleanupFloat = () => {{
    try {{
      if (floating && floatHost) floatHost.remove();
    }} catch (e) {{}}
  }};
  window.addEventListener('unload', cleanupFloat);
  window.addEventListener('pagehide', cleanupFloat);
  try {{ if (_dbg) {{ _dbg.style.display = 'none'; }} }} catch (e) {{}}
  }} catch (err) {{
    _showErr('<div style="background:rgba(202,0,0,0.95);color:#fff;padding:10px 14px;border-radius:8px;font:13px/1.5 monospace;">'
      + '<b>❌ 组件 JS 运行错误</b><br/>' + (err && err.message ? err.message : err) + '<br/>'
      + '<pre style="font-size:10px;white-space:pre-wrap;margin:4px 0;">' + (err && err.stack ? err.stack.substring(0, 500) : '') + '</pre></div>');
  }}
}})();
</script>
"""

    component_height = 920 if debug_mode else 640
    components.html(html, height=component_height, scrolling=False)


_default_data_dir = str(Path(__file__).resolve().parents[3] / "data")
if "stream_url_input" not in st.session_state:
    # 首次进入默认指向项目根目录下的真实视频（绝对路径，不受启动目录影响）
    _default_videos = list_video_files(_default_data_dir)
    st.session_state["stream_url_input"] = _default_videos[0] if _default_videos else "data/demo_video.mp4"
if "video_data_dir" not in st.session_state:
    st.session_state["video_data_dir"] = _default_data_dir
if "task_id" not in st.session_state:
    st.session_state["task_id"] = None
if "pending_task_id" not in st.session_state:
    st.session_state["pending_task_id"] = None
if "pending_stream_url" not in st.session_state:
    st.session_state["pending_stream_url"] = ""
if "health" not in st.session_state:
    st.session_state["health"] = None
if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = "native"
if "recognition_logs" not in st.session_state:
    st.session_state["recognition_logs"] = []
if "task_duration" not in st.session_state:
    st.session_state["task_duration"] = 0
if "align_offset_frames" not in st.session_state:
    st.session_state["align_offset_frames"] = 8
if "task_stopped" not in st.session_state:
    st.session_state["task_stopped"] = False
if "debug_mode" not in st.session_state:
    st.session_state["debug_mode"] = False


def get_current_stream_url() -> str:
    value = st.session_state.get("stream_url_input", "")
    return value.strip() if isinstance(value, str) else ""

st.sidebar.header("控制台")

with st.sidebar.expander("连接配置", expanded=False):
    # 默认 API 地址可用环境变量覆盖（远程访问时设为 http://<服务器IP>:8000，免手填）
    api_base = st.text_input("API Base", os.getenv("CSA_API_BASE", "http://127.0.0.1:8000"))
    st.caption(f"当前 API: {api_base}")

with st.sidebar.expander("视频源", expanded=True):
    video_data_dir = st.text_input("视频目录", key="video_data_dir")
    video_options = list_video_files(video_data_dir)
    if video_options:
        selected_video = st.selectbox(
            "选择本地视频",
            options=video_options,
            format_func=lambda p: os.path.relpath(p, video_data_dir),
        )
        if st.button("使用所选视频"):
            st.session_state["stream_url_input"] = selected_video
            existing_pending_task = st.session_state.get("pending_task_id")
            existing_pending_stream = st.session_state.get("pending_stream_url", "")
            if existing_pending_task and existing_pending_stream == selected_video:
                st.info("已存在该视频的后台任务，将直接复用。")
            else:
                selected_duration = int(st.session_state.get("task_duration", 0))
                response = requests.post(
                    f"{api_base}/api/v1/analyze/stream",
                    json={"stream_url": selected_video, "duration": selected_duration, "callback_url": None},
                    timeout=10,
                )
                if response.ok:
                    data = response.json()
                    st.session_state["pending_task_id"] = data["task_id"]
                    st.session_state["pending_stream_url"] = selected_video
                    st.session_state["task_id"] = None
                    st.session_state["task_stopped"] = False
                    st.session_state["ui_mode"] = "native"
                    st.session_state["ui_mode_radio"] = "native"
                else:
                    st.session_state["pending_task_id"] = None
                    st.session_state["pending_stream_url"] = ""
                    st.error(f"后台任务启动失败: {response.text}")
    else:
        st.info("未扫描到可用视频文件")
    stream_url = st.text_input("Stream URL", key="stream_url_input")
    st.caption(f"当前输入源: {summarize_stream_source(stream_url)}")

# 低频设置合并折叠：推理档位 + 任务参数
with st.sidebar.expander("高级设置", expanded=False):
    st.caption("推理档位")
    profile_options = ["realtime", "high_accuracy"]
    selected_profile = st.selectbox("推理档位", options=profile_options, index=0)
    apply_profile_clicked = st.button("应用档位", use_container_width=True)
    if apply_profile_clicked:
        profile_resp = requests.post(f"{api_base}/api/v1/profile", params={"profile": selected_profile}, timeout=10)
        if profile_resp.ok:
            st.success(f"档位已切换为: {selected_profile}")
        else:
            st.error(f"档位切换失败: {profile_resp.text}")

    st.divider()
    st.caption("任务参数")
    duration = st.number_input("Duration(s, 0=无限制)", value=int(st.session_state.get("task_duration", 0)), min_value=0)
    st.session_state["task_duration"] = int(duration)
    align_offset_frames = st.slider(
        "框前瞻偏移(帧)",
        min_value=0,
        max_value=10,
        value=int(st.session_state.get("align_offset_frames", 8)),
        step=1,
        help="检测结果相对视频帧的前瞻量：用于遮盖结果到达延迟。0=严格对齐当前帧（结果延迟时会看到框落后）。",
    )
    st.session_state["align_offset_frames"] = int(align_offset_frames)
    debug_mode = st.checkbox(
        "调试模式",
        value=bool(st.session_state.get("debug_mode", False)),
        help="勾选后在“可视化分析”界面下方显示实时参数，便于排查性能和结果。",
    )
    st.session_state["debug_mode"] = bool(debug_mode)

# 任务操作常驻侧边栏底部：状态 + 开始/停止
st.sidebar.divider()
current_task_id = st.session_state.get("task_id")
if st.session_state.get("task_stopped") and not current_task_id:
    st.sidebar.caption("当前任务: 已停止")
else:
    st.sidebar.caption(f"当前任务: {current_task_id or '未创建'}")
start_clicked = st.sidebar.button("▶ 开始分析", use_container_width=True, type="primary")
stop_clicked = st.sidebar.button("⏹ 停止分析", use_container_width=True)

if start_clicked:
    pending_task_id = st.session_state.get("pending_task_id")
    pending_stream_url = st.session_state.get("pending_stream_url", "")
    current_stream_url = get_current_stream_url()
    if not pending_task_id:
        st.error("请先点击“使用所选视频”，后台启动任务后再点击“开始分析”。")
    elif pending_stream_url and current_stream_url and pending_stream_url != current_stream_url:
        st.error("当前输入源与后台预启动任务不一致，请重新点击“使用所选视频”。")
    else:
        st.session_state["task_id"] = pending_task_id
        st.session_state["task_stopped"] = False
        st.session_state["ui_mode"] = "native"
        st.session_state["ui_mode_radio"] = "native"
        st.success(f"当前任务已绑定: {pending_task_id}")

if stop_clicked:
    active_task_id = st.session_state.get("task_id") or st.session_state.get("pending_task_id")
    if not active_task_id:
        st.session_state["ui_mode"] = "idle"
        st.session_state["task_stopped"] = True
    else:
        try:
            stop_resp = requests.post(f"{api_base}/api/v1/tasks/{active_task_id}/stop", timeout=10)
            if not stop_resp.ok:
                st.warning(f"停止任务失败: {stop_resp.text}")
        except Exception as exc:
            st.warning(f"停止任务请求异常: {exc}")

        st.session_state["ui_mode"] = "idle"
        st.session_state["task_stopped"] = True
        st.session_state["task_id"] = None
        st.session_state["pending_task_id"] = None
        st.session_state["pending_stream_url"] = ""

left, right = st.columns([2, 1])

task_id = st.session_state.get("task_id")

with left:
    st.subheader("🎬 视频与可视化")
    c_mode, c_stop = st.columns([5, 1])
    with c_mode:
        if st.session_state.get("ui_mode_radio") not in MODE_OPTIONS:
            st.session_state["ui_mode_radio"] = "native"
        st.radio(
            "视图模式",
            MODE_OPTIONS,
            format_func=lambda x: MODE_LABELS[x],
            horizontal=True,
            key="ui_mode_radio",
            on_change=_on_ui_mode_change,
            label_visibility="collapsed",
        )
    with c_stop:
        if st.button("⏹ 停止", use_container_width=True, help="停止可视化覆盖层（不停止后台任务）"):
            st.session_state["ui_mode"] = "idle"

    mode = st.session_state.get("ui_mode", "native")
    current_stream_url = get_current_stream_url()
    media_url = build_media_url(api_base, current_stream_url)

    if mode == "native":
        if current_stream_url.startswith("http://") or current_stream_url.startswith("https://") or (Path(current_stream_url).exists() and Path(current_stream_url).is_file()):
            st.video(current_stream_url)
            st.info("当前模式: 原生视频播放")
        else:
            st.info("当前视频源不可用于原生播放，请检查路径或 URL。")
    elif mode == "visualize":
        active_task_id = st.session_state.get("task_id")
        if not active_task_id:
            st.info("尚未启动任务，请先点击“开始分析”，再点击“可视化分析”。")
        elif not media_url:
            st.info("当前视频源无法映射为浏览器可访问地址，无法进行覆盖层可视化。")
        else:
            task_snapshot: dict = {}
            try:
                status_resp = requests.get(f"{api_base}/api/v1/tasks/{active_task_id}", timeout=10)
                if status_resp.ok:
                    task_snapshot = status_resp.json() if isinstance(status_resp.json(), dict) else {}
                    stream_stats = task_snapshot.get("stream_stats", {}) if isinstance(task_snapshot, dict) else {}
                    pushed = int(stream_stats.get("pushed_frames", 0) or 0)
                    read = int(stream_stats.get("read_frames", 0) or 0)
                    if read > 0:
                        st.caption(f"实时处理进度(快照): pushed={pushed} / read={read}（覆盖层状态栏显示实时 pushed/read）")
            except Exception:
                pass

            ws_url = build_ws_url(api_base, active_task_id)
            render_overlay_component(
                media_url=media_url,
                ws_url=ws_url,
                align_offset_frames=int(st.session_state.get("align_offset_frames", 8)),
                debug_mode=bool(st.session_state.get("debug_mode", False)),
                task_status=str(task_snapshot.get("status", "unknown")) if isinstance(task_snapshot, dict) else "unknown",
                task_error=str(task_snapshot.get("error", "")) if isinstance(task_snapshot, dict) else "",
                stream_stats=task_snapshot.get("stream_stats", {}) if isinstance(task_snapshot, dict) else {},
                api_base=api_base,
            )
    elif mode == "zones":
        st.subheader("🎯 布控区域绘制")
        st.caption("截取当前视频画面，鼠标点击添加多边形顶点，定义禁区/危险区域")
        import cv2 as _cv2
        _stream = get_current_stream_url()
        _vpath = Path(_stream) if not _stream.startswith("http") else None

        c_z1, c_z2 = st.columns([1, 1])
        if c_z1.button("📸 截取当前帧", use_container_width=True):
            if _vpath and _vpath.exists():
                cap = _cv2.VideoCapture(str(_vpath))
                cap.set(_cv2.CAP_PROP_POS_FRAMES, 100)
                ok, fr = cap.read()
                cap.release()
                if ok:
                    st.session_state["zone_frame"] = fr
                    st.session_state["zone_points"] = []
                else:
                    st.warning("无法读取视频帧")
            else:
                st.warning("请先选择一个本地视频文件")

        zone_frame = st.session_state.get("zone_frame")
        zone_points = st.session_state.get("zone_points", [])

        if zone_frame is not None:
            from streamlit_image_coordinates import streamlit_image_coordinates
            _draw = zone_frame.copy()
            for pt in zone_points:
                _cv2.circle(_draw, tuple(pt), 5, (0, 200, 0), -1)
            if len(zone_points) >= 2:
                for i in range(1, len(zone_points)):
                    _cv2.line(_draw, tuple(zone_points[i-1]), tuple(zone_points[i]), (0, 200, 0), 2)
            if len(zone_points) >= 3:
                _cv2.line(_draw, tuple(zone_points[-1]), tuple(zone_points[0]), (0, 200, 255), 1)

            _h, _w = _draw.shape[:2]
            _disp_w = 560
            _disp_h = int(_h * _disp_w / _w)
            _disp = _cv2.resize(_draw, (_disp_w, _disp_h))
            _scale = _w / _disp_w

            click = streamlit_image_coordinates(_disp, height=_disp_h)
            if click is not None and isinstance(click, dict) and "x" in click:
                cx = int(click["x"] * _scale)
                cy = int(click["y"] * _scale)
                new_pt = [cx, cy]
                if not zone_points or new_pt != zone_points[-1]:
                    zone_points.append(new_pt)
                    st.session_state["zone_points"] = zone_points
                    st.rerun()

            st.caption(f"已点击 {len(zone_points)} 个顶点" + ("（至少 3 个才能构成多边形）" if len(zone_points) < 3 else ""))

            c_z3, c_z4 = st.columns([1, 1])
            if c_z3.button("↩️ 撤销上一点", use_container_width=True) and zone_points:
                zone_points.pop()
                st.session_state["zone_points"] = zone_points
                st.rerun()
            if c_z4.button("🗑️ 清空", use_container_width=True) and zone_points:
                st.session_state["zone_points"] = []
                st.rerun()

            zone_name = st.text_input("区域名称", value="", placeholder="例如: 左侧禁区", key="zone_name_input")
            zone_type = st.selectbox("区域类型", ["forbidden", "danger"],
                                     format_func=lambda x: "禁区（S02 闯入）" if x == "forbidden" else "危险区（H04 靠近）",
                                     key="zone_type_sel")
            if st.button("💾 保存区域", use_container_width=True) and len(zone_points) >= 3 and zone_name.strip():
                resp = requests.post(f"{api_base}/api/v1/zones", json={
                    "name": zone_name.strip(),
                    "zone_type": zone_type,
                    "polygon": zone_points,
                }, timeout=10)
                if resp.ok:
                    st.success(f"✅ 区域已保存并生效: {resp.json().get('zone_id')}")
                    st.session_state["zone_points"] = []
                    st.session_state.pop("zone_frame", None)
                else:
                    st.error(f"保存失败: {resp.text[:150]}")
            elif len(zone_points) < 3:
                st.caption("至少需要 3 个顶点")

        st.subheader("已有布控区域")
        try:
            zones_resp = requests.get(f"{api_base}/api/v1/zones", timeout=10).json()
            for ztype, zlist in zones_resp.items():
                for z in zlist:
                    zid = z.get("id", "")
                    npts = len(z.get("polygon", []))
                    cza, czb = st.columns([3, 1])
                    cza.markdown(f"🔹 `{zid}` ({ztype.replace('_zones','')}, {npts} 顶点)")
                    if czb.button("删除", key=f"del_{zid}", use_container_width=True):
                        requests.delete(f"{api_base}/api/v1/zones/{zid}", timeout=10)
                        st.rerun()
            if not any(zones_resp.values()):
                st.info("暂无布控区域。截取帧面后画多边形保存。")
        except Exception as exc:
            st.caption(f"读取区域失败: {exc}")

    else:
        st.info("已停止可视化。点击下方按钮恢复上次视图。")
        if st.button("▶ 恢复可视化", type="primary"):
            _on_ui_mode_change()
            st.rerun()

with right:
    render_scene_panel(api_base, task_id)
    render_dashboard(api_base)
    with st.expander("🔍 场景检索（本地知识库）", expanded=False):
        render_search(api_base)

# 管理类功能折叠到全宽页签，主区只保留“视频 + 态势概要”
tab_rules, tab_logs, tab_health = st.tabs(["📜 告警规则", "📋 识别日志与导出", "🩺 系统状态"])
with tab_rules:
    render_rules(api_base)
with tab_logs:
    render_logs(api_base)
with tab_health:
    render_health(api_base, task_id)
