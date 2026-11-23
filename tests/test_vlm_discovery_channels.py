"""发现器双通道（M01/M02）单元测试。

运行：python tests/test_vlm_discovery_channels.py
（无 pytest 依赖；记录文件写入临时目录，退出自动清理）
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.vlm_discovery import DISCOVERY_PROMPT, M01_PROMPT, M02_PROMPT, VLMDiscoveryScheduler  # noqa: E402
from src.pipeline.inference_engine import InferenceEngine  # noqa: E402


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, sec: float) -> None:
        self.now += sec


class FakeReviewer:
    enabled = True

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.prompts: list[str] = []

    def review_clip(self, frames, track_id="", prompt=""):
        self.prompts.append(prompt)
        label = self.labels.pop(0) if self.labels else "Normal"
        return {"status": "success", "label": label, "confidence": 0.8,
                "description": "stub", "latency_ms": 5}


BASE_CFG = {
    "enabled": True, "periodic_interval_sec": 90, "per_track_cooldown_sec": 30,
    "budget_per_hour": 0, "burst": 8, "max_inflight": 3,
    "min_track_age_frames": 0, "min_track_score": 0.0, "low_conf_threshold": 0.4,
    "m01_enabled": True, "m01_cooldown_sec": 2.5, "m01_window_gap_sec": 2.0,
    "m01_confirm_cooldown_sec": 30.0, "m01_require_dual_window": True,
    "m02_enabled": True, "m02_cooldown_sec": 2.5, "m02_window_gap_sec": 2.0,
    "m02_confirm_cooldown_sec": 30.0, "m02_require_dual_window": True,
}


def wait_records(path: Path, n: int, timeout: float = 2.0) -> list[dict]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) >= n:
                return rows
        time.sleep(0.05)
    return []


def run_channel_case(channel: str, target: str, other_label: str) -> dict:
    """返回该通道的 first/second/rejected/prompt 断言结果。"""
    res = {}
    tmpdir = tempfile.TemporaryDirectory()
    record = Path(tmpdir.name) / "events.jsonl"
    clock = FakeClock()
    reviewer = FakeReviewer()
    sched = VLMDiscoveryScheduler(BASE_CFG, reviewer=reviewer, record_path=str(record), clock=clock)
    frame = [np.zeros((64, 64, 3), np.uint8)]

    reviewer.labels = [target]
    p1 = sched.maybe_propose("t1", 100, target, 0.6, source="rule_pending",
                             video_time=10.0, channel=channel)
    res["proposed"] = p1 is not None and p1.get("channel") == channel
    sched.submit(p1, frame)
    rows = wait_records(record, 1)
    res["first"] = bool(rows) and rows[0].get("window") == "first" and rows[0].get("confirmed") is False
    res["prompt"] = bool(reviewer.prompts) and reviewer.prompts[0] == (M01_PROMPT if channel == "m01" else M02_PROMPT)

    # 间隔不足（冷却已过，但视频时间只推进 1.5s < gap 2.0）
    clock.advance(3.0)
    reviewer.labels = [target]
    p2 = sched.maybe_propose("t1", 120, target, 0.6, source="rule_pending",
                             video_time=11.5, channel=channel)
    res["too_soon_proposed"] = p2 is not None
    if p2 is not None:
        sched.submit(p2, frame)
    rows = wait_records(record, 2)
    res["too_soon"] = bool(rows) and rows[-1].get("window") == "too_soon"

    # 间隔足够 → 双窗确认
    clock.advance(3.0)
    reviewer.labels = [target]
    p3 = sched.maybe_propose("t1", 160, target, 0.6, source="rule_pending",
                             video_time=16.0, channel=channel)
    res["second_proposed"] = p3 is not None
    if p3 is not None:
        sched.submit(p3, frame)
    rows = wait_records(record, 3)
    res["confirmed"] = bool(rows) and rows[-1].get("window") == "second" and rows[-1].get("confirmed") is True

    # 确认后长冷却
    clock.advance(1.0)
    p4 = sched.maybe_propose("t1", 170, target, 0.6, source="rule_pending",
                             video_time=16.0, channel=channel)
    res["post_cooldown"] = p4 is None

    # 另一通道不受影响（同一 track）：m01 已确认不应阻塞 m02 的 first
    other_channel = "m02" if channel == "m01" else "m01"
    clock.advance(1.0)
    reviewer.labels = [other_label]
    p5 = sched.maybe_propose("t1", 180, other_label, 0.6, source="rule_pending",
                             video_time=17.0, channel=other_channel)
    res["cross_channel_independent"] = p5 is not None
    if p5 is not None:
        sched.submit(p5, frame)
        rows = wait_records(record, 4)
        res["cross_channel_first"] = bool(rows) and rows[-1].get("window") == "first"

    tmpdir.cleanup()
    return res


def test_generic_prompt_and_frames() -> dict:
    res = {}
    tmpdir = tempfile.TemporaryDirectory()
    record = Path(tmpdir.name) / "events.jsonl"
    clock = FakeClock()
    reviewer = FakeReviewer()
    sched = VLMDiscoveryScheduler(BASE_CFG, reviewer=reviewer, record_path=str(record), clock=clock)
    reviewer.labels = ["Normal"]
    p = sched.maybe_propose("t9", 300, "Normal", 0.3, source="pc_gray", video_time=30.0)
    sched.submit(p, [np.zeros((32, 32, 3), np.uint8)])
    wait_records(record, 1)
    res["generic_prompt"] = reviewer.prompts[-1] == DISCOVERY_PROMPT

    class Buf:
        def recent_entries(self, k):
            return [(1, np.full((720, 1280, 3), 50, np.uint8)),
                    (2, np.full((720, 1280, 3), 80, np.uint8))]

    class FakeEngine:
        frame_buffer = Buf()

    frames = InferenceEngine._build_crop_discovery_frames(
        FakeEngine(), {"track_id": "t1", "bbox": [100, 100, 160, 240]},
        np.full((720, 1280, 3), 120, np.uint8), 1024)
    res["two_crop_frames"] = len(frames) == 2
    res["max_side_1024"] = all(max(f.shape[:2]) == 1024 for f in frames)
    res["no_pano"] = all(f.shape[:2] != (720, 1280) for f in frames)

    class Buf3:
        def recent_entries(self, k):
            return [(1, np.zeros((72, 128, 3), np.uint8)),
                    (2, np.full((72, 128, 3), 200, np.uint8)),
                    (3, np.full((72, 128, 3), 10, np.uint8))]

    class FakeEngine3:
        frame_buffer = Buf3()
        _crop_resize = staticmethod(InferenceEngine._crop_resize)
        _peak_frame = staticmethod(InferenceEngine._peak_frame)

    hframes = InferenceEngine._build_h_single_frames(
        FakeEngine3(), {"track_id": "t", "bbox": [10, 10, 40, 50]},
        np.zeros((72, 128, 3), np.uint8), 1024)
    res["h_single_frames"] = len(hframes) == 3
    res["peak_frame_picked"] = InferenceEngine._peak_frame(
        Buf3().recent_entries(8)) is not None
    tmpdir.cleanup()
    return res


def test_channel_quota() -> dict:
    """M01 通道配额耗尽时，M02 仍应有自己的一份（不互相饿死）。"""
    res = {}
    tmpdir = tempfile.TemporaryDirectory()
    record = Path(tmpdir.name) / "events.jsonl"
    clock = FakeClock()
    reviewer = FakeReviewer()
    cfg = dict(BASE_CFG)
    cfg.update({"budget_per_hour": 0, "m01_burst": 1, "m01_budget_per_hour": 1,
                "m02_burst": 1, "m02_budget_per_hour": 120})
    sched = VLMDiscoveryScheduler(cfg, reviewer=reviewer, record_path=str(record), clock=clock)

    reviewer.labels = ["M01"]
    p1 = sched.maybe_propose("t1", 100, "M01", 0.6, source="rule_pending",
                             video_time=10.0, channel="m01")
    res["m01_first_ok"] = p1 is not None
    if p1 is not None:
        sched.submit(p1, [np.zeros((32, 32, 3), np.uint8)])
        wait_records(record, 1)

    clock.advance(3.0)
    p2 = sched.maybe_propose("t1", 130, "M01", 0.6, source="rule_pending",
                             video_time=13.0, channel="m01")
    res["m01_quota_blocks"] = p2 is None

    reviewer.labels = ["M02"]
    p3 = sched.maybe_propose("t1", 130, "M02", 0.6, source="rule_pending",
                             video_time=13.0, channel="m02")
    res["m02_not_starved"] = p3 is not None
    tmpdir.cleanup()
    return res


def test_vote_channel() -> dict:
    """H 类 2-of-3 投票：3 次里命中 2 次确认；只命中 1 次拒绝。"""
    res = {}
    cfg = dict(BASE_CFG)
    cfg.update({"budget_per_hour": 0, "h_single_cooldown_sec": 0.5,
                "h_single_burst": 5, "h_single_budget_per_hour": 120,
                "h_single_vote_window_sec": 10, "h_single_vote_min_hits": 2})
    frame = [np.zeros((32, 32, 3), np.uint8)]
    for name, labels, expect in (
        ("two_hits", ["H03", "Normal", "H03"], True),
        ("one_hit", ["H03", "Normal", "Normal"], False),
    ):
        tmpdir = tempfile.TemporaryDirectory()
        record = Path(tmpdir.name) / "events.jsonl"
        clock = FakeClock()
        reviewer = FakeReviewer()
        sched = VLMDiscoveryScheduler(cfg, reviewer=reviewer, record_path=str(record), clock=clock)
        for i, label in enumerate(labels):
            clock.advance(1.0)
            reviewer.labels = [label]
            p = sched.maybe_propose("t1", 100 + i, "H03", 0.6, source="rule_pending",
                                    video_time=10.0 + i * 2, channel="h_single")
            if p is not None:
                sched.submit(p, frame)
                wait_records(record, i + 1)
        rows = wait_records(record, len(labels))
        last = rows[-1] if rows else {}
        if expect:
            res[name] = last.get("window") == "vote_confirmed" and last.get("confirmed") is True
        else:
            res[name] = last.get("window") == "vote_rejected" and last.get("confirmed") is False
        tmpdir.cleanup()
    return res


def test_pc_high_h_trigger() -> dict:
    """PC 高置信 H 类也应送审（P0 结构性漏检）。"""
    res = {}
    tmpdir = tempfile.TemporaryDirectory()
    clock = FakeClock()
    reviewer = FakeReviewer()
    sched = VLMDiscoveryScheduler(BASE_CFG, reviewer=reviewer,
                                  record_path=str(Path(tmpdir.name) / "e.jsonl"), clock=clock)
    p = sched.maybe_propose("t1", 100, "H03", 0.88, source="pc_high",
                            video_time=10.0, channel="h_single")
    res["pc_high_h_proposed"] = p is not None
    tmpdir.cleanup()
    return res


def test_h_pair_frames_and_aggregation() -> dict:
    """双人帧构建（全景双框+两人裁剪）与 pair 冷却聚合。"""
    res = {}
    class Buf:
        def recent_entries(self, k):
            return [(1, np.full((720, 1280, 3), 50, np.uint8))]

    class FakeEngine:
        frame_buffer = Buf()
        _crop_resize = staticmethod(InferenceEngine._crop_resize)

    frames = InferenceEngine._build_h_pair_frames(
        FakeEngine(), {"track_id": "a", "bbox": [100, 100, 160, 240]},
        [400, 120, 180, 260], np.full((720, 1280, 3), 120, np.uint8), 1024)
    res["h_pair_frame_count"] = len(frames) == 3
    res["h_pair_crops_1024"] = all(max(f.shape[:2]) == 1024 for f in frames[1:])

    # pair 冷却：同一 pair 4s 内第二次提案被挡
    tmpdir = tempfile.TemporaryDirectory()
    clock = FakeClock()
    reviewer = FakeReviewer()
    cfg = dict(BASE_CFG)
    cfg.update({"budget_per_hour": 0, "h_pair_cooldown_sec": 4.0,
                "h_pair_burst": 5, "h_pair_budget_per_hour": 120})
    sched = VLMDiscoveryScheduler(cfg, reviewer=reviewer,
                                  record_path=str(Path(tmpdir.name) / "e.jsonl"), clock=clock)
    p1 = sched.maybe_propose("a:b", 100, "H01", 0.8, source="rule_pending",
                             video_time=10.0, channel="h_pair")
    clock.advance(1.0)
    p2 = sched.maybe_propose("a:b", 120, "H01", 0.8, source="rule_pending",
                             video_time=11.0, channel="h_pair")
    res["pair_cooldown_blocks"] = p1 is not None and p2 is None
    tmpdir.cleanup()
    return res


def test_geom_veto_two_strikes() -> dict:
    """强几何 H03：同一缓存只计一票，两次不同结论才允许压制。"""
    class FakeEngine:
        def __init__(self):
            self._geom_veto_last_cache = {}
            self._geom_veto_streak = {}

        _geom_veto_allowed = InferenceEngine._geom_veto_allowed

    res = {}
    eng = FakeEngine()
    res["first_vote_no_suppress"] = eng._geom_veto_allowed("t1", 100, True) is False
    res["same_cache_not_counted_twice"] = eng._geom_veto_allowed("t1", 100, True) is False
    res["second_distinct_vote_suppresses"] = eng._geom_veto_allowed("t1", 130, True) is True
    res["non_strong_single_vote_ok"] = FakeEngine()._geom_veto_allowed("t2", 100, False) is True
    return res


def main() -> int:
    results = {}
    results.update({f"m01_{k}": v for k, v in run_channel_case("m01", "M01", "M02").items()})
    results.update({f"m02_{k}": v for k, v in run_channel_case("m02", "M02", "M01").items()})
    results.update(test_generic_prompt_and_frames())
    results.update({f"quota_{k}": v for k, v in test_channel_quota().items()})
    results.update({f"vote_{k}": v for k, v in test_vote_channel().items()})
    results.update(test_pc_high_h_trigger())
    results.update({f"hpair_{k}": v for k, v in test_h_pair_frames_and_aggregation().items()})
    results.update({f"geomveto_{k}": v for k, v in test_geom_veto_two_strikes().items()})

    print(json.dumps(results, ensure_ascii=False, indent=1))
    failed = [k for k, v in results.items() if not v]
    print("FAILED:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
