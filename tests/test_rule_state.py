"""规则状态归一单测：step/bump/arm/tick 语义 + StateRegistry 清理。"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.behavior_reasoner import BehaviorReasoner  # noqa: E402
from src.core.rule_state import StateRegistry, active, arm, bump, step, tick  # noqa: E402


def test_primitives() -> dict:
    res = {}
    table: dict[str, int] = defaultdict(int)
    res["step_up"] = step(table, "a", True) == 1 and step(table, "a", True) == 2
    res["step_down_floor"] = step(table, "a", False) == 1 and step(table, "a", False) == 0 \
        and step(table, "a", False) == 0
    res["step_custom"] = step(table, "b", True, up=2) == 2 and step(table, "b", False, down=2) == 0
    res["bump"] = bump(table, "c", 2) == 2 and bump(table, "c", -5) == 0
    res["arm_keeps_max"] = arm(table, "h", 5) == 5 and arm(table, "h", 3) == 5
    tick(table, "h")
    res["tick"] = table["h"] == 4 and active(table, "h")
    for _ in range(5):
        tick(table, "h")
    res["tick_floor_inactive"] = table["h"] == 0 and not active(table, "h")
    return res


def test_registry_cleanup() -> dict:
    res = {}
    registry = StateRegistry()
    tracks = registry.track(defaultdict(int))
    pairs = registry.pair(defaultdict(int))
    tracks.update({"a": 1, "b": 2})
    pairs.update({"a:b": 3, "b:c": 4})
    registry.drop_stale({"a"})
    res["track_dropped"] = set(tracks) == {"a"}
    res["pair_dropped"] = set(pairs) == set()  # 任一方失效则整对清理
    return res


def test_reasoner_registry() -> dict:
    """规则引擎所有 per-track/pair 表都注册到同一个 registry。"""
    res = {}
    reasoner = BehaviorReasoner({})
    stats = reasoner._state.stats()
    res["registry_has_tables"] = stats["track_tables"] >= 20 and stats["pair_tables"] >= 6

    reasoner.track_phone_counter["P1"] = 3
    reasoner.track_h03_counter["P1"] = 2
    reasoner.pair_h01_counter["P1:P2"] = 4
    reasoner.pair_last_frame["P1:P2"] = 10
    reasoner.track_first_frame_id["P1"] = 1
    reasoner.clear_stale_tracks({"P2"})
    res["track_cleared"] = "P1" not in reasoner.track_phone_counter \
        and "P1" not in reasoner.track_h03_counter \
        and "P1" not in reasoner.track_first_frame_id
    res["pair_cleared"] = "P1:P2" not in reasoner.pair_h01_counter \
        and "P1:P2" not in reasoner.pair_last_frame
    return res


def test_floor_polygon_geometry() -> dict:
    """地面多边形：点在多边形内/外；空多边形回退地面线。"""
    res = {}
    reasoner = BehaviorReasoner({})
    poly = [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]
    reasoner.update_zones(floor_polygon=poly)
    res["polygon_inside"] = reasoner._bottom_on_floor([100, 500, 200, 300], (1000, 1000))
    res["polygon_outside"] = not reasoner._bottom_on_floor([100, 100, 200, 100], (1000, 1000))
    reasoner.update_zones(floor_polygon=[])
    reasoner.update_zones(floor_line_ratio=0.90)
    res["line_fallback"] = reasoner._bottom_on_floor([0, 850, 100, 100], (1000, 1000)) \
        and not reasoner._bottom_on_floor([0, 700, 100, 100], (1000, 1000))
    return res


def main() -> int:
    results = {}
    results.update({f"prim_{k}": v for k, v in test_primitives().items()})
    results.update({f"reg_{k}": v for k, v in test_registry_cleanup().items()})
    results.update({f"reasoner_{k}": v for k, v in test_reasoner_registry().items()})
    results.update({f"floor_{k}": v for k, v in test_floor_polygon_geometry().items()})
    import json
    print(json.dumps(results, ensure_ascii=False, indent=1))
    failed = [k for k, v in results.items() if not v]
    print("FAILED:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
