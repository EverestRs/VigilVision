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
        self.record_path = Path(record_path)
        self.cache_lookup = cache_lookup
        self._clock = clock or time.monotonic

        self._lock = threading.Lock()
        self._tokens = self.burst
        self._last_refill = self._clock()
        self._last_submit: dict[str, float] = {}
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
    ) -> dict[str, Any] | None:
        """判断当前 track 是否值得送 VLM 发现器；返回 proposal 或 None。"""
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

            last = self._last_submit.get(track_id, -1e9)
            if now - last < self.per_track_cooldown_sec:
                self._stats["skipped_cooldown"] += 1
                return None

            low_conf = float(confidence) < self.low_conf_threshold
            rule_pending = str(source).startswith("rule_pending")
            periodic_due = now - self._last_periodic.get(track_id, -1e9) >= self.periodic_interval_sec
            if not (low_conf or rule_pending or periodic_due):
                return None

            if image_hash is not None:
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

            self._last_submit[track_id] = now
            if periodic_due:
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
                result = self.reviewer.review_clip(
                    frames, track_id=str(proposal["track_id"]), prompt=DISCOVERY_PROMPT)
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
                self._append_record(proposal, result)
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
                })
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

    def _append_record(self, proposal: dict[str, Any], result: dict[str, Any]) -> None:
        try:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                **proposal,
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
            "record_path": str(self.record_path),
        }
