"""自动地面标定：流模式采样逻辑单测（用本地文件模拟流，不依赖 RTSP）。"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.floor_calibration import calibrate_from_video, collect_points  # noqa: E402


class FakeDet:
    def __init__(self, empty=False):
        self.empty = empty

    def infer(self, img):
        if self.empty:
            return []
        return [{"bbox": [20, 30, 40, 80], "score": 0.9}]


class FakePose:
    def infer(self, crop, meta):
        kps = [[0.0, 0.0, 0.0] for _ in range(133)]
        # 踝点（相对 crop）：y=60 → 绝对 y=90，落在下半区
        kps[15] = [10.0, 60.0, 0.9]
        kps[16] = [12.0, 60.0, 0.9]
        return kps


def make_video(path: Path, frames: int = 20, size=(160, 120), fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 120 + i, np.uint8)
        writer.write(img)
    writer.release()


def main() -> int:
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "fake_stream.mp4"
        make_video(video)
        det, pose = FakeDet(), FakePose()

        points, used = collect_points(video, det, pose, None, sample_sec=0.1,
                                      max_frames=8, is_stream=True, duration_sec=10)
        res["stream_frames"] = used == 8
        res["stream_points"] = len(points) > 0

        points_f, used_f = collect_points(video, det, pose, None, sample_sec=0.1,
                                          max_frames=8, is_stream=False)
        res["file_frames"] = used_f == 8
        res["file_points"] = len(points_f) > 0

        summary = calibrate_from_video(video, FakeDet(empty=True), pose, None,
                                       floor_line=0.90, max_frames=5,
                                       is_stream=True, duration_sec=5)
        res["empty_fallback_band"] = summary.get("result") == "insufficient_points_fallback_band" \
            and len(summary.get("polygon") or []) == 4
        res["stream_mode_reported"] = summary.get("source_mode") == "stream"

    import json
    print(json.dumps(res, ensure_ascii=False, indent=1))
    failed = [k for k, v in res.items() if not v]
    print("FAILED:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
