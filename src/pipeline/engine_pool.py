"""多相机引擎池（接口预留）。

单路/默认相机行为不变：`engine` 仍指向默认引擎。多路时按 camera_id 懒创建，
每个引擎独立持有 frame_buffer / 规则状态 / 叙述器，避免 `begin_task` 互相清空。

显存是硬约束：默认最多 2 个引擎，超过时返回明确错误（由 API 转 429），
不做静默淘汰，避免把仍在跑的任务引擎关掉。
"""
from __future__ import annotations

import threading
from typing import Callable

from src.pipeline.inference_engine import InferenceEngine


class EnginePool:
    def __init__(self, factory: Callable[[str], InferenceEngine], max_engines: int = 2) -> None:
        self._factory = factory
        self.max_engines = max(1, int(max_engines))
        self._engines: dict[str, InferenceEngine] = {}
        self._lock = threading.RLock()

    def get(self, camera_id: str = "default") -> InferenceEngine:
        key = str(camera_id or "default")
        with self._lock:
            engine = self._engines.get(key)
            if engine is not None:
                return engine
            if len(self._engines) >= self.max_engines:
                raise RuntimeError(
                    f"引擎池已满（{len(self._engines)}/{self.max_engines}），"
                    f"无法为相机 {key} 创建新引擎")
            engine = self._factory(key)
            self._engines[key] = engine
            return engine

    def can_accept(self, camera_id: str = "default") -> bool:
        with self._lock:
            key = str(camera_id or "default")
            return key in self._engines or len(self._engines) < self.max_engines

    def set(self, camera_id: str, engine: InferenceEngine) -> None:
        with self._lock:
            self._engines[str(camera_id or "default")] = engine

    def get_if_exists(self, camera_id: str = "default") -> InferenceEngine | None:
        with self._lock:
            return self._engines.get(str(camera_id or "default"))

    def close(self, camera_id: str) -> None:
        with self._lock:
            engine = self._engines.pop(str(camera_id or "default"), None)
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass

    def close_all(self) -> None:
        with self._lock:
            engines = list(self._engines.values())
            self._engines.clear()
        for engine in engines:
            try:
                engine.close()
            except Exception:
                pass

    def status(self) -> dict:
        with self._lock:
            return {
                "max_engines": self.max_engines,
                "count": len(self._engines),
                "cameras": sorted(self._engines.keys()),
            }
