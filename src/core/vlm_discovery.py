"""VLM 发现器调度（第一期）：事件触发 + 周期采样，预算队列与单 track 冷却。

设计目标：
- 与主推理循环解耦：`maybe_propose` 只做轻量策略判断，真正的 VLM 调用在后台线程执行；
- 控制成本：令牌桶（每小时预算）+ 单 track 冷却 + 全局在飞上限 + 连续失败退避；
- 产出可复用：每次调用的结论落盘到 JSONL，供离线复核/训练数据挖掘使用；
- 不改变主判定链路：发现结论不直接覆盖 PC/规则标签（避免形成反馈环），
  是否升级为告警由后续阶段按验收结果决定。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable

DISCOVERY_PROMPT = (
    "你是校园安全视频的巡查员。请观察这几帧画面（可能包含同一场景的前后时刻），"
    "判断是否存在需要关注的行为或事件。\n"
    "候选类别：\n"
    "- Normal：正常通行、站立、坐下、蹲下、正常交谈；\n"
    "- H01 打架斗殴, H02 校园霸凌, H03 跌倒/昏厥, H04 疑似轻生, H05 破坏设施；\n"
    "- M01 吸烟, M02 使用手机, M03 遮挡摄像头, S01 异常徘徊, S02 翻越围栏。\n"
    "判定原则：\n"
    "1) 只依据画面中可见证据，不做猜测；证据不足时输出 Normal；\n"
    "2) 办公/会议场景手部靠近面部（托腮、写字、喝水）不算吸烟；\n"
    "3) 坐下、蹲下、弯腰不算跌倒；跌倒需人体横卧/贴近地面且无法自行起身。\n"
    "先给一句不超过 40 字的画面描述，再给结论。仅输出 JSON，不要输出其他文字：\n"
    '{"description": "...", "label": "Normal", "confidence": 0.8}'
)


# 离线挖掘专用 prompt：比运行时更"逐人排查"，允许把可疑姿态说出来由人工复核；
# 运行时仍用 DISCOVERY_PROMPT（更保守，避免自动告警误报）。
MINING_PROMPT = (
    "你是校园安全视频的复核员。第一张是全景（红框标出画面中的目标人物），"
    "其余是画面中不同人物的局部放大（可能不止一人，也可能少于红框数）。\n"
    "请逐人观察并描述可见姿态/动作，只要**任意一人**满足下列任一行为就按其类别输出：\n"
    "- H03 跌倒/昏厥：人体横卧、贴近地面、坐/躺在地上无法起身；\n"
    "- H01 打架斗殴、H02 校园霸凌（推搡/殴打/多人围堵）；\n"
    "- H04 疑似轻生（攀爬栏杆/窗台、悬坐边缘）；H05 破坏设施（踢打/砸物品）；\n"
    "- M01 吸烟：手指间夹持细长物举向嘴部、有烟雾；办公场景托腮/写字/喝水不算；\n"
    "- M02 使用手机、M03 遮挡摄像头、S01 异常徘徊、S02 翻越围栏；\n"
    "- Normal：站立、行走、坐下、蹲下、正常交谈，无上述行为。\n"
    "证据不足时判 Normal，不要猜测。仅输出 JSON，不要其他文字：\n"
    '{"description": "...", "label": "Normal", "confidence": 0.8}'
)


# M01 专用通道：实测（2026-11-22）"仅人物裁剪 + 定向 prompt + 两帧"在真实吸烟片段
# 5/5 命中、负样本 1/4 误报；而"全景+裁剪+通用 11 类 prompt"四种分辨率组合全部 0/5。
# 关键不是分辨率，而是：不要让全景稀释注意力，并且直接问"是否吸烟、看烟支/烟雾"。
M01_PROMPT = (
    "这些图片是同一个人的连续放大帧（均为人物裁剪）。判断此人是否正在吸烟：\n"
    "1) 手指间是否夹持细长烟支（白色/深色细杆），并靠近嘴部；\n"
    "2) 嘴边或面部是否有烟雾；3) 是否有吸食/吐烟动作。\n"
    "托腮、摸脸、拿手机、拿笔写字、喝水/饮料都不算吸烟。\n"
    "只依据可见证据，看不清时判 Normal。仅输出 JSON，不要其他文字：\n"
    '{"description": "...", "label": "M01或Normal", "confidence": 0.8}'
)


# M02 专用通道：与 M01 同构（"手持物"类），裁剪-only + 定向 prompt + 双窗。
# 手机在裁剪中比全景清晰得多，且"手在脸附近"的托腮/喝水同样需要被排除。
M02_PROMPT = (
    "这些图片是同一个人的连续放大帧（均为人物裁剪）。判断此人是否正在使用手机：\n"
    "1) 手中是否握持手机（矩形屏幕/机身）；2) 屏幕是否朝向面部；\n"
    "3) 是否低头看屏、拇指滑动/点击，且持续一段时间。\n"
    "拿纸、拿笔、喝水、托腮、摸脸、整理头发都不算使用手机。\n"
    "只依据可见证据，看不清时判 Normal。仅输出 JSON，不要其他文字：\n"
    '{"description": "...", "label": "M02或Normal", "confidence": 0.8}'
)


# H 类通道：需要全景/双人上下文。
# h_pair：打架/霸凌——全景（双框）+ 两人裁剪；判定"是否互相推搡/挥拳/围堵"，区分玩闹。
H_PAIR_PROMPT = (
    "第一张是全景（红框标出两名当事人），其余两张分别是这两人的放大裁剪。"
    "判断这两人之间是否存在：H01 打架（互相挥拳/推搡/扭打）、H02 霸凌（一方被围堵/殴打/不敢反抗）。"
    "需要看到双向的身体接触或攻击意图；追逐、握手、拍肩、玩闹式打闹判 Normal。"
    "仅输出 JSON，不要其他文字："
    '{"description": "...", "label": "H01或H02或Normal", "confidence": 0.8}'
)

# h_single：跌倒/轻生/破坏——全景提供地面、栏杆、物体等场景参照，裁剪补充姿态细节。
H_SINGLE_PROMPT = (
    "第一张是全景（红框标出目标人物），其余为该人物/场景的放大帧。判断是否存在：\n"
    "- H03 跌倒/昏厥：人体横卧、贴近地面、无法自行起身（坐下、蹲下、弯腰不算）；\n"
    "- H04 疑似轻生：攀爬栏杆/窗台、悬坐边缘、做出翻越动作；\n"
    "- H05 破坏设施：踢打、砸、掀翻物品，且能看到物体状态改变。\n"
    "证据不足判 Normal。仅输出 JSON，不要其他文字："
    '{"description": "...", "label": "H03或H04或H05或Normal", "confidence": 0.8}'
)


class VLMDiscoveryScheduler:
    """发现器策略与调度器（不持有模型，调用注入的 reviewer）。"""

    def __init__(
        self,
        config: dict[str, Any] | None,
        reviewer: Any,
        record_path: str = "results/discovery/discovery_events.jsonl",
        cache_lookup: Callable[[Any], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.reviewer = reviewer
        self.periodic_interval_sec = float(cfg.get("periodic_interval_sec", 90))
        self.per_track_cooldown_sec = float(cfg.get("per_track_cooldown_sec", 30))
        self.budget_per_hour = float(cfg.get("budget_per_hour", 240))
        self.burst = float(cfg.get("burst", 4))
        self.max_inflight = int(cfg.get("max_inflight", 3))
        self.min_track_age_frames = int(cfg.get("min_track_age_frames", 24))
        self.min_track_score = float(cfg.get("min_track_score", 0.35))
        self.low_conf_threshold = float(cfg.get("low_conf_threshold", 0.40))
        # 手持物专用通道（M01 吸烟 / M02 手机）：裁剪-only + 定向 prompt + 双窗时序一致性。
        # 两个通道共用一套机制，仅 prompt/阈值不同；H 类仍走通用通道（全景+历史帧）。
        self.channels: dict[str, dict[str, Any]] = {
            "m01": {
                "enabled": bool(cfg.get("m01_enabled", True)),
                "image_max_px": int(cfg.get("m01_image_max_px", 1024)),
                "cooldown": float(cfg.get("m01_cooldown_sec", 2.5)),
                "gap": float(cfg.get("m01_window_gap_sec", 2.0)),
                "confirm_cooldown": float(cfg.get("m01_confirm_cooldown_sec", 30.0)),
                "require_dual": bool(cfg.get("m01_require_dual_window", True)),
                "prompt": M01_PROMPT,
            },
            "m02": {
                "enabled": bool(cfg.get("m02_enabled", True)),
                "image_max_px": int(cfg.get("m02_image_max_px", 1024)),
                "cooldown": float(cfg.get("m02_cooldown_sec", 2.5)),
                "gap": float(cfg.get("m02_window_gap_sec", 2.0)),
                "confirm_cooldown": float(cfg.get("m02_confirm_cooldown_sec", 30.0)),
                "require_dual": bool(cfg.get("m02_require_dual_window", True)),
                "prompt": M02_PROMPT,
            },
            # H01/H02：双人事件，按 pair 聚合（proposal 的 track_id 用 pair_key）
            "h_pair": {
                "enabled": bool(cfg.get("h_pair_enabled", True)),
                "image_max_px": int(cfg.get("h_pair_image_max_px", 1024)),
                "cooldown": float(cfg.get("h_pair_cooldown_sec", 4.0)),
                "confirm_cooldown": float(cfg.get("h_pair_confirm_cooldown_sec", 20.0)),
                "mode": "vote",
                "vote_window_sec": float(cfg.get("h_pair_vote_window_sec", 10.0)),
                "vote_min_hits": int(cfg.get("h_pair_vote_min_hits", 2)),
                "prompt": H_PAIR_PROMPT,
            },
            # H03/H04/H05：单人/场景事件，全景+裁剪；2-of-3 投票
            "h_single": {
                "enabled": bool(cfg.get("h_single_enabled", True)),
                "image_max_px": int(cfg.get("h_single_image_max_px", 1024)),
                "cooldown": float(cfg.get("h_single_cooldown_sec", 4.0)),
                "confirm_cooldown": float(cfg.get("h_single_confirm_cooldown_sec", 20.0)),
                "mode": "vote",
                "vote_window_sec": float(cfg.get("h_single_vote_window_sec", 10.0)),
                "vote_min_hits": int(cfg.get("h_single_vote_min_hits", 2)),
                "prompt": H_SINGLE_PROMPT,
            },
        }
        # m01/m02 仍是连续双窗；H 类走 2-of-3 投票
        for _name in ("m01", "m02"):
            self.channels[_name].setdefault("mode", "dual")
        self._channel_targets = {
            "m01": {"M01"},
            "m02": {"M02"},
            "h_pair": {"H01", "H02"},
            "h_single": {"H03", "H04", "H05"},
        }
        # 每通配额：避免 M01 把全局预算吃光、M02/通用通道被饿死
        # （实测 40s 内 M01 连发 11 次，M02 一次都没轮到）
        self._channel_tokens: dict[str, float] = {}
        self._channel_burst: dict[str, float] = {}
        self._channel_budget: dict[str, float] = {}
        self._channel_last_refill: dict[str, float] = {}
        for name, chan in self.channels.items():
            burst = float(cfg.get(f"{name}_burst", 3))
            self._channel_burst[name] = burst
            self._channel_tokens[name] = burst
            self._channel_budget[name] = float(cfg.get(f"{name}_budget_per_hour", 180))
            self._channel_last_refill[name] = (clock or time.monotonic)()
        self.record_path = Path(record_path)
        self.cache_lookup = cache_lookup
        self._clock = clock or time.monotonic

        # RLock：maybe_propose 持有锁时会再调用 _take_channel_token/_refund_token
        self._lock = threading.RLock()
        self._tokens = self.burst
        self._last_refill = self._clock()
        self._last_submit: dict[str, float] = {}
        self._channel_last_submit: dict[tuple[str, str], float] = {}
        self._channel_confirmed: dict[tuple[str, str], float] = {}
        self._channel_first: dict[tuple[str, str], dict[str, Any]] = {}
        self._channel_votes: dict[tuple[str, str], deque] = {}
        self._channel_stats: dict[str, dict[str, int]] = {
            name: {"first": 0, "confirmed": 0, "rejected": 0, "votes": 0} for name in self.channels
        }
        self._last_periodic: dict[str, float] = {}
        self._first_seen: dict[str, int] = {}
        self._last_hash: dict[str, tuple[int, float]] = {}
        self._inflight = 0
        self._fail_streak = 0
        self._disabled_until = 0.0
        self._last_error = ""
        self._stats = {
            "proposed": 0,
            "submitted": 0,
            "skipped_cooldown": 0,
            "skipped_budget": 0,
            "skipped_inflight": 0,
            "skipped_dedupe": 0,
            "skipped_track": 0,
            "ok": 0,
            "error": 0,
            "dual_first": 0,
            "dual_confirmed": 0,
            "dual_rejected": 0,
        }

    # ── 预算 ──────────────────────────────────────────────────────
    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        if self.budget_per_hour > 0:
            self._tokens = min(self.burst, self._tokens + elapsed * self.budget_per_hour / 3600.0)

    def _take_token(self) -> bool:
        self._refill()
        if self.budget_per_hour <= 0:
            return True  # 0 表示不限预算（仅用于调试）
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _refund_token(self) -> None:
        with self._lock:
            self._tokens = min(self.burst, self._tokens + 1.0)

    def _take_channel_token(self, name: str) -> bool:
        """通道级令牌桶：保证 M01/M02 各有独立配额，互不饿死。"""
        with self._lock:
            tokens = self._channel_tokens.get(name, 0.0)
            burst = self._channel_burst.get(name, 0.0)
            budget = self._channel_budget.get(name, 0.0)
            now = self._clock()
            elapsed = max(0.0, now - self._channel_last_refill.get(name, now))
            self._channel_last_refill[name] = now
            if budget > 0:
                tokens = min(burst, tokens + elapsed * budget / 3600.0)
            self._channel_tokens[name] = tokens
            if budget <= 0 or tokens >= 1.0:
                self._channel_tokens[name] = max(0.0, tokens - 1.0)
                return True
            return False

    # ── 策略 ──────────────────────────────────────────────────────
    def maybe_propose(
        self,
        track_id: str,
        frame_id: int,
        label: str,
        confidence: float,
        source: str = "",
        track_score: float = 1.0,
        video_time: float = 0.0,
        image_hash: int | None = None,
        channel: str = "generic",
    ) -> dict[str, Any] | None:
        """判断当前 track 是否值得送 VLM 发现器；返回 proposal 或 None。

        channel="m01" 时走专用通道：crop-only + 定向 prompt + 双窗一致性，
        冷却/去重规则与通用通道独立。
        """
        if not self.enabled or not self.reviewer or not getattr(self.reviewer, "enabled", True):
            return None
        now = self._clock()
        if now < self._disabled_until:
            return None

        with self._lock:
            self._stats["proposed"] += 1
            first = self._first_seen.setdefault(track_id, int(frame_id))
            age = int(frame_id) - first
            if age < self.min_track_age_frames or float(track_score) < self.min_track_score:
                self._stats["skipped_track"] += 1
                return None

            chan_name = str(channel)
            chan = self.channels.get(chan_name)
            is_dual = chan is not None and bool(chan.get("enabled"))
            key = (track_id, chan_name)
            if is_dual:
                # 确认后进入长冷却，避免同一段行为反复确认
                if now - self._channel_confirmed.get(key, -1e9) < float(chan["confirm_cooldown"]):
                    self._stats["skipped_cooldown"] += 1
                    return None
                last = self._channel_last_submit.get(key, -1e9)
                cooldown = float(chan["cooldown"])
            else:
                last = self._last_submit.get(track_id, -1e9)
                cooldown = self.per_track_cooldown_sec
            if now - last < cooldown:
                self._stats["skipped_cooldown"] += 1
                return None

            low_conf = float(confidence) < self.low_conf_threshold
            rule_pending = str(source).startswith("rule_pending")
            periodic_due = now - self._last_periodic.get(track_id, -1e9) >= self.periodic_interval_sec
            # M01 通道不额外放宽触发条件：只在本就有可疑信号（rule_pending/低置信/周期）时送审，
            # 否则每个 M01 标签帧都会送一次，几秒内就能把预算吃光（实测 40s 内 11 次）。
            # P0：PC 高置信的 H 类也送审（只进发现器/待确认，不直接告警），
            # 解决"分类器看对了但 VLM 从没机会确认"的结构性漏检。
            pc_high_h = str(source).startswith("pc_high") and str(label).startswith("H")
            if not (low_conf or rule_pending or periodic_due or pc_high_h):
                return None

            # dHash 去重只对通用通道生效：M01 双窗需要的是"同一目标不同时刻"，画面本就相似
            if image_hash is not None and not is_dual:
                prev = self._last_hash.get(track_id)
                if prev is not None and now - prev[1] < self.per_track_cooldown_sec and self._hamming(prev[0], int(image_hash)) <= 2:
                    self._stats["skipped_dedupe"] += 1
                    return None
                self._last_hash[track_id] = (int(image_hash), now)

            if self._inflight >= self.max_inflight:
                self._stats["skipped_inflight"] += 1
                return None
            if not self._take_token():
                self._stats["skipped_budget"] += 1
                return None
            if is_dual and not self._take_channel_token(chan_name):
                self._refund_token()
                self._stats["skipped_budget"] += 1
                return None

            self._last_submit[track_id] = now
            if is_dual:
                self._channel_last_submit[key] = now
            if periodic_due and not is_dual:
                self._last_periodic[track_id] = now
            self._inflight += 1
            self._stats["submitted"] += 1
            reasons = []
            if low_conf:
                reasons.append("low_conf")
            if rule_pending:
                reasons.append("rule_pending")
            if periodic_due:
                reasons.append("periodic")
            return {
                "proposal_id": uuid.uuid4().hex[:12],
                "track_id": track_id,
                "frame_id": int(frame_id),
                "video_time": round(float(video_time), 3),
                "claimed_label": str(label),
                "claimed_conf": round(float(confidence), 3),
                "source": str(source),
                "reasons": reasons,
                "channel": chan_name if is_dual else "generic",
                "ts": round(time.time(), 3),
            }

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return bin(int(a) ^ int(b)).count("1")

    # ── 执行 ──────────────────────────────────────────────────────
    def submit(self, proposal: dict[str, Any], frames: list[Any],
               on_result: Callable[[dict[str, Any], dict[str, Any]], None] | None = None) -> bool:
        """后台线程执行一次 VLM 发现调用；失败会退避并归还预算。"""
        if not frames:
            self._release(proposal, refund=True)
            return False

        def _run() -> None:
            result: dict[str, Any] = {"status": "error", "label": "Normal", "confidence": 0.0}
            try:
                chan = self.channels.get(str(proposal.get("channel", "generic")))
                prompt = str(chan.get("prompt", DISCOVERY_PROMPT)) if chan else DISCOVERY_PROMPT
                result = self.reviewer.review_clip(
                    frames, track_id=str(proposal["track_id"]), prompt=prompt)
                ok = str(result.get("status", "")) == "success"
                with self._lock:
                    self._stats["ok" if ok else "error"] += 1
                    if ok:
                        self._fail_streak = 0
                    else:
                        self._fail_streak += 1
                        self._last_error = str(result.get("reasoning", ""))[:160]
                        if self._fail_streak >= 3:
                            self._disabled_until = self._clock() + 120.0
                extra = self._consistency(proposal, result)
                self._append_record(proposal, result, extra=extra)
                if on_result is not None:
                    try:
                        on_result(result, proposal)
                    except Exception:
                        pass
            except Exception as exc:
                with self._lock:
                    self._stats["error"] += 1
                    self._fail_streak += 1
                    self._last_error = f"discovery_failed: {exc}"[:160]
                    if self._fail_streak >= 3:
                        self._disabled_until = self._clock() + 120.0
                self._append_record(proposal, {
                    "status": "error", "label": "Normal", "confidence": 0.0,
                    "reasoning": f"discovery_failed: {exc}",
                }, extra={"channel": proposal.get("channel", "generic")})
            finally:
                self._release(proposal, refund=False)

        threading.Thread(target=_run, name=f"vlm-discovery-{proposal['proposal_id']}", daemon=True).start()
        return True

    def cancel(self, proposal: dict[str, Any]) -> None:
        """提交失败/无帧可送时归还预算与在飞额度。"""
        self._release(proposal, refund=True)

    def _release(self, proposal: dict[str, Any], refund: bool) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            if refund:
                self._stats["submitted"] = max(0, self._stats["submitted"] - 1)
                self._last_submit.pop(str(proposal.get("track_id", "")), None)
        if refund:
            self._refund_token()

    def _consistency(self, proposal: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """通道级一致性：m01/m02 连续双窗；h_pair/h_single 走 2-of-3 投票。"""
        channel = str(proposal.get("channel", "generic"))
        chan = self.channels.get(channel)
        fields: dict[str, Any] = {"channel": channel}
        if chan is None or not bool(chan.get("enabled")):
            return fields

        track_id = str(proposal.get("track_id", ""))
        verdict = str(result.get("label", "Normal"))
        video_time = float(proposal.get("video_time", 0.0) or 0.0)
        now = self._clock()
        key = (track_id, channel)
        targets = self._channel_targets.get(channel, set())
        channel_stats = self._channel_stats.setdefault(
            channel, {"first": 0, "confirmed": 0, "rejected": 0, "votes": 0})

        if str(chan.get("mode", "dual")) == "vote":
            window = float(chan.get("vote_window_sec", 10.0))
            min_hits = int(chan.get("vote_min_hits", 2))
            stamp = video_time if video_time > 0 else now
            with self._lock:
                votes = self._channel_votes.setdefault(key, deque(maxlen=3))
                votes.append((stamp, verdict in targets))
                while votes and stamp - votes[0][0] > window:
                    votes.popleft()
                hits = sum(1 for _, hit in votes if hit)
                channel_stats["votes"] += 1
                fields["votes"] = f"{hits}/{len(votes)}"
                if hits >= min_hits and len(votes) >= 2:
                    self._channel_votes.pop(key, None)
                    self._channel_confirmed[key] = now
                    self._stats["dual_confirmed"] += 1
                    channel_stats["confirmed"] += 1
                    fields.update({"window": "vote_confirmed", "confirmed": True})
                elif len(votes) >= 3:
                    self._channel_votes.pop(key, None)
                    self._stats["dual_rejected"] += 1
                    channel_stats["rejected"] += 1
                    fields.update({"window": "vote_rejected", "confirmed": False})
                else:
                    fields.update({"window": "vote_pending", "confirmed": False})
            return fields

        target = next(iter(targets), "")
        with self._lock:
            first = self._channel_first.get(key)
            if verdict == target:
                if first is None:
                    if not bool(chan.get("require_dual")):
                        self._channel_confirmed[key] = now
                        self._stats["dual_confirmed"] += 1
                        channel_stats["confirmed"] += 1
                        fields.update({"window": "single", "confirmed": True})
                    else:
                        self._channel_first[key] = {
                            "video_time": video_time,
                            "ts": now,
                            "proposal_id": proposal.get("proposal_id", ""),
                        }
                        self._stats["dual_first"] += 1
                        channel_stats["first"] += 1
                        fields.update({"window": "first", "confirmed": False})
                else:
                    gap = (video_time - float(first.get("video_time", 0.0))
                           if video_time > 0 else now - float(first.get("ts", now)))
                    if gap >= float(chan.get("gap", 2.0)):
                        self._channel_first.pop(key, None)
                        self._channel_confirmed[key] = now
                        self._stats["dual_confirmed"] += 1
                        channel_stats["confirmed"] += 1
                        fields.update({"window": "second", "confirmed": True})
                    else:
                        fields.update({"window": "too_soon", "confirmed": False})
            else:
                if first is not None:
                    self._channel_first.pop(key, None)
                    self._stats["dual_rejected"] += 1
                    channel_stats["rejected"] += 1
                fields.update({"window": "rejected", "confirmed": False})
        return fields

    def _append_record(self, proposal: dict[str, Any], result: dict[str, Any],
                       extra: dict[str, Any] | None = None) -> None:
        try:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                **proposal,
                **(extra or {}),
                "verdict": str(result.get("label", "Normal")),
                "verdict_conf": round(float(result.get("confidence", 0.0)), 3),
                "description": str(result.get("description", ""))[:300],
                "latency_ms": int(result.get("latency_ms", 0) or 0),
                "status": str(result.get("status", "error")),
            }
            with self.record_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── 观测 ──────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        now = self._clock()
        return {
            "enabled": self.enabled,
            "tokens": round(self._tokens, 2),
            "burst": self.burst,
            "inflight": self._inflight,
            "max_inflight": self.max_inflight,
            "fail_streak": self._fail_streak,
            "disabled_for_sec": max(0.0, round(self._disabled_until - now, 1)),
            "last_error": self._last_error,
            "stats": dict(self._stats),
            "channels": {
                name: {
                    "enabled": bool(chan.get("enabled")),
                    "cooldown": chan.get("cooldown"),
                    "window_gap": chan.get("gap"),
                    "require_dual": chan.get("require_dual"),
                    "tokens": round(self._channel_tokens.get(name, 0.0), 2),
                    "budget_per_hour": self._channel_budget.get(name, 0.0),
                    **self._channel_stats.get(name, {}),
                }
                for name, chan in self.channels.items()
            },
            "dual_pending_first": len(self._channel_first),
            "record_path": str(self.record_path),
        }
