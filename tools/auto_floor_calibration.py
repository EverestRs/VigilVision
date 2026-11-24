"""自动地面多边形标定 CLI（核心逻辑在 src/core/floor_calibration.py）。

用法：
  python tools/auto_floor_calibration.py --video ../data/laying_corner_1-1.mp4 \
      --floor-line 0.90 --out results/calibration/floor_auto.json
  # 直接写回后端：
  python tools/auto_floor_calibration.py --video ../data/xxx.mp4 --apply-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.behavior_reasoner import BehaviorReasoner  # noqa: E402
from src.core.detector import Detector  # noqa: E402
from src.core.floor_calibration import calibrate_from_video  # noqa: E402
from src.core.pose_estimator import PoseEstimator  # noqa: E402


def build(cfg: dict, device: str):
    runtime = dict(cfg.get("runtime", {}))
    if device == "cpu":
        runtime["execution_provider"] = "CPUExecutionProvider"
        runtime["device"] = "cpu"
    det = Detector(dict(cfg.get("detector", {}), runtime=dict(runtime)))
    pose_cfg = dict(cfg.get("pose", {}))
    pose_cfg["runtime"] = dict(runtime)
    pose = PoseEstimator(pose_cfg)
    reasoner = BehaviorReasoner(dict(cfg.get("behavior_heuristics", {})))
    return det, pose, reasoner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="本地视频文件")
    ap.add_argument("--rtsp", default="", help="RTSP/RTMP/HTTP 流地址")
    ap.add_argument("--duration-sec", type=float, default=30.0,
                    help="流模式最长采样时长")
    ap.add_argument("--sample-sec", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out", default="results/calibration/floor_polygon_auto.json")
    ap.add_argument("--apply-url", default="")
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--floor-line", type=float, default=0.90)
    args = ap.parse_args()

    source = (args.rtsp or args.video).strip()
    if not source:
        print("[auto-floor] 需要 --video 或 --rtsp")
        return 2
    is_stream = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    video = Path(source)
    if not is_stream and not video.is_absolute():
        video = (ROOT / video).resolve()
    if is_stream:
        video = source
    cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    det, pose, reasoner = build(cfg, args.device)
    summary = calibrate_from_video(video, det, pose, reasoner,
                                   floor_line=args.floor_line,
                                   sample_sec=args.sample_sec,
                                   max_frames=args.max_frames,
                                   min_points=args.min_points,
                                   is_stream=is_stream,
                                   duration_sec=(args.duration_sec if is_stream else None))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 诊断图：仅文件模式（流已消费，不重连）
    try:
        if is_stream:
            raise RuntimeError("stream_skip_debug")
        cap = cv2.VideoCapture(str(video))
        ok, img = cap.read()
        cap.release()
        if ok:
            h, w = img.shape[:2]
            poly = summary["polygon"]
            poly_px = (np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
                       if poly else None)
            if poly_px is not None:
                cv2.polylines(img, [poly_px], True, (0, 0, 255), 3)
            debug_path = out_path.with_suffix(".jpg")
            cv2.imwrite(str(debug_path), cv2.resize(img, (1280, int(1280 * h / w))))
            summary["debug_image"] = str(debug_path.relative_to(ROOT))
    except Exception:
        pass

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    if args.apply_url:
        import requests
        resp = requests.post(f"{args.apply_url.rstrip('/')}/api/v1/camera/floor-polygon",
                             json={"points": summary["polygon"]}, timeout=15)
        print("[auto-floor] apply", resp.status_code, resp.text[:160])
        if not resp.ok:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
