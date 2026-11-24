"""Le2i 跌倒评测：逐帧几何 → 视频级判定（H03 几何先验）。

数据：seanphan/le2i-sentinel-frames（每视频 3 帧，320x240，GT 在 metadata/ground_truth.json）
指标：
- 视频级 recall：has_fall=True 的视频中，任一帧触发 fall_like 的比例
- 视频级 false positive：has_fall=False 的视频中，任一帧触发 fall_like 的比例
- 另统计 refine_light 是否产出 H03 规则候选（3 帧通常不足以形成时序，仅作参考）

用法：
  python tools/eval_h03_le2i.py --frames-dir ../data_external/public/le2i_frames/frames \
      --gt ../data_external/public/le2i_frames/metadata/ground_truth.json \
      --out results/eval/h03_le2i.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.behavior_reasoner import BehaviorReasoner  # noqa: E402
from src.core.detector import Detector  # noqa: E402
from src.core.pose_estimator import PoseEstimator  # noqa: E402


def build(cfg: dict, device: str):
    runtime = dict(cfg.get("runtime", {}))
    if device == "cpu":
        runtime["execution_provider"] = "CPUExecutionProvider"
        runtime["device"] = "cpu"
    det = Detector(dict(cfg.get("detector", {}), runtime=dict(runtime)))
    pose_cfg = dict(cfg.get("pose", {}))
    pose_cfg["onnx_path"] = cfg.get("pose", {}).get("onnx_path", "models/pose/vitpose_wholebody_s_int8.onnx")
    pose_cfg["runtime"] = dict(runtime)
    pose = PoseEstimator(pose_cfg)
    reasoner = BehaviorReasoner(dict(cfg.get("behavior_heuristics", {})))
    return det, pose, reasoner


def largest_person(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: float(d.get("score", 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="../data_external/public/le2i_frames/frames")
    ap.add_argument("--gt", default="../data_external/public/le2i_frames/metadata/ground_truth.json")
    ap.add_argument("--out", default="results/eval/h03_le2i.json")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--frames-per-video", type=int, default=3)
    ap.add_argument("--floor-line", type=float, default=None,
                    help="覆盖机位地面线（归一化 y），用于验证不同机位标定")
    ap.add_argument("--floor-polygon", default=None,
                    help="覆盖地面多边形，JSON 字符串，如 '[[0.02,0.52],[0.98,0.52],[0.98,1.0]]'")
    args = ap.parse_args()

    root = Path(args.frames_dir)
    if not root.is_absolute():
        root = (ROOT / args.frames_dir).resolve()
    gt_path = Path(args.gt)
    if not gt_path.is_absolute():
        gt_path = (ROOT / args.gt).resolve()
    gt = {row["vid_id"]: bool(row.get("has_fall")) for row in json.loads(gt_path.read_text(encoding="utf-8"))}

    model_cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    det, pose, reasoner = build(model_cfg, args.device)
    if args.floor_line is not None:
        reasoner.update_zones(floor_line_ratio=float(args.floor_line))
    if args.floor_polygon:
        reasoner.update_zones(floor_polygon=json.loads(args.floor_polygon))

    videos = sorted([d for d in root.iterdir() if d.is_dir()])
    if args.limit_videos > 0:
        videos = videos[:args.limit_videos]

    rows = []
    for vid in videos:
        frames = sorted(vid.glob("*.jpg"))[:args.frames_per_video]
        if not frames:
            continue
        frame_results = []
        for fid, path in enumerate(frames, start=1):
            img = cv2.imread(str(path))
            if img is None:
                continue
            dets = sorted(det.infer(img) or [], key=lambda d: float(d.get("score", 0.0)), reverse=True)[:3]
            persons = []
            for idx, person in enumerate(dets):
                bbox = [float(v) for v in person.get("bbox", [0, 0, 0, 0])]
                x, y, w, h = [int(v) for v in bbox]
                crop = img[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0:
                    continue
                kps = pose.infer(crop, {"bbox": [x, y, w, h]})
                fall_like = bool(reasoner._is_fall_like(bbox, img.shape[:2], kps))
                posture = bool(reasoner._fall_posture_candidate(bbox, img.shape[:2], kps))
                # 走一遍主路径规则（逐人独立 track；3 帧只作参考）
                label, conf, src, tags = reasoner.refine_light(
                    track_id=f"{vid.name}_{idx}", bbox=bbox, keypoints=kps,
                    frame_shape=img.shape[:2], base_label="Normal", base_confidence=0.6,
                    peer_tracks=[], frame_id=fid)
                persons.append({
                    "bbox": [round(v, 1) for v in bbox], "fall_like": fall_like,
                    "posture": posture, "label": label, "source": src, "tags": list(tags),
                })
            frame_results.append({
                "frame": path.name,
                "person": bool(persons),
                "fall_like": any(p["fall_like"] for p in persons),
                "posture": any(p["posture"] for p in persons),
                "persons": persons,
            })
        person_frames = [f for f in frame_results if f.get("person")]
        any_fall = any(f.get("fall_like") for f in person_frames)
        any_posture = any(f.get("posture") for f in person_frames)
        any_h03 = any("fall_geometry_rule" in (f.get("tags") or []) or f.get("label") == "H03"
                      for f in person_frames)
        rows.append({
            "vid_id": vid.name,
            "has_fall": bool(gt.get(vid.name, False)),
            "frames": len(frames),
            "person_frames": len(person_frames),
            "any_fall_like": any_fall,
            "any_posture": any_posture,
            "any_h03_rule": any_h03,
            "details": frame_results,
        })

    def rate(flag: str, truth: bool) -> tuple[int, int, float]:
        subset = [r for r in rows if r["has_fall"] is truth]
        hit = sum(1 for r in subset if r[flag])
        return hit, len(subset), round(hit / len(subset), 4) if subset else 0.0

    summary = {
        "videos": len(rows),
        "fall_videos": sum(1 for r in rows if r["has_fall"]),
        "normal_videos": sum(1 for r in rows if not r["has_fall"]),
        "fall_recall_by_fall_like": rate("any_fall_like", True),
        "normal_fp_by_fall_like": rate("any_fall_like", False),
        "fall_recall_by_posture": rate("any_posture", True),
        "normal_fp_by_posture": rate("any_posture", False),
        "fall_recall_by_h03_rule": rate("any_h03_rule", True),
        "normal_fp_by_h03_rule": rate("any_h03_rule", False),
        "no_person_videos": sum(1 for r in rows if r["person_frames"] == 0),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
