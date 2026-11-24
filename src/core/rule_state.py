"""规则引擎状态归一：统一的 per-track / per-pair 状态表与计数语义。

背景：behavior_reasoner 里长期存在 20+ 个 `defaultdict(int)` 计数器，各自用
`+= 1` / `max(0, -1)` / 手写保持窗口；清理逻辑也是手写清单（曾出现 pair_key
分隔符不一致导致状态永不清理的 bug）。这里把三件事收敛成一套：

1. `StateRegistry`：所有 per-track / per-pair 表统一注册，`drop_stale()` 一次清理；
2. `step()`：统一的“命中 +up / 未命中衰减 down”计数语义；
3. `arm()` / `tick()`：统一的保持窗口（hold）语义。

表仍是 dict，不改变数据形态，便于逐步迁移、可用 golden 回归逐帧比对。
"""
from __future__ import annotations

from typing import Any


def step(counter: dict[str, int], key: str, hit: bool, up: int = 1, down: int = 1) -> int:
    """计数一步：命中加 up，未命中减 down（下限 0）。"""
    value = int(counter.get(key, 0))
    value = value + up if hit else max(0, value - down)
    counter[key] = value
    return value


def bump(counter: dict[str, int], key: str, amount: int = 1, floor: int = 0) -> int:
    """无命中语义的增减（保留旧代码里的 +2 / -1 等特殊步长）。"""
    value = max(floor, int(counter.get(key, 0)) + int(amount))
    counter[key] = value
    return value


def arm(counter: dict[str, int], key: str, frames: int) -> int:
    """把保持窗口设为至少 frames（取 max，避免短窗口覆盖长窗口）。"""
    value = max(int(counter.get(key, 0)), int(frames))
    counter[key] = value
    return value


def tick(counter: dict[str, int], key: str, down: int = 1) -> int:
    """窗口/冷却推进一帧。"""
    value = max(0, int(counter.get(key, 0)) - int(down))
    counter[key] = value
    return value


def active(counter: dict[str, int], key: str) -> bool:
    return int(counter.get(key, 0)) > 0


class StateRegistry:
    """统一注册与清理 per-track / per-pair 状态表。"""

    def __init__(self) -> None:
        self._track_tables: list[dict[Any, Any]] = []
        self._pair_tables: list[dict[Any, Any]] = []

    def track(self, table: dict[Any, Any]) -> dict[Any, Any]:
        self._track_tables.append(table)
        return table

    def pair(self, table: dict[Any, Any]) -> dict[Any, Any]:
        self._pair_tables.append(table)
        return table

    def drop_stale(self, active_track_ids: set[str], pair_sep: str = ":") -> None:
        """清理不在活跃集合中的 track 状态；pair 键按 `a:b` 拆分判定。"""
        for table in self._track_tables:
            stale = [key for key in list(table.keys()) if key not in active_track_ids]
            for key in stale:
                table.pop(key, None)
        for table in self._pair_tables:
            stale = [key for key in list(table.keys())
                     if any(part not in active_track_ids for part in str(key).split(pair_sep))]
            for key in stale:
                table.pop(key, None)

    def stats(self) -> dict[str, int]:
        return {
            "track_tables": len(self._track_tables),
            "pair_tables": len(self._pair_tables),
            "track_entries": sum(len(t) for t in self._track_tables),
            "pair_entries": sum(len(t) for t in self._pair_tables),
        }
