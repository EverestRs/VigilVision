"""EnginePool 单测：复用、上限、关闭。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.engine_pool import EnginePool  # noqa: E402


class FakeEngine:
    def __init__(self, cid):
        self.cid = cid
        self.closed = False

    def close(self):
        self.closed = True


def main() -> int:
    res = {}
    made = []

    def factory(cid):
        made.append(cid)
        return FakeEngine(cid)

    pool = EnginePool(factory, max_engines=2)
    a1 = pool.get("a")
    a2 = pool.get("a")
    b = pool.get("b")
    res["same_camera_reuses"] = a1 is a2 and len(made) == 2
    try:
        pool.get("c")
        res["max_engines_enforced"] = False
    except RuntimeError:
        res["max_engines_enforced"] = True
    res["status_cameras"] = pool.status()["count"] == 2
    pool.close("a")
    res["close_marks_engine"] = a1.closed and pool.get_if_exists("a") is None
    res["other_engine_alive"] = not b.closed
    res["can_create_after_close"] = pool.get("c").cid == "c"

    import json
    print(json.dumps(res, ensure_ascii=False, indent=1))
    failed = [k for k, v in res.items() if not v]
    print("FAILED:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
