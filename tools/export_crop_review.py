"""导出 M01 候选帧的人工复核清单（crop 图 + CSV）。

用途：Step 2 抽样核对 crop 提案/规则门控后的确认精度。
流程：检测/跟踪/姿态/PC/规则/crop（含提案与门控）→ 收集最终 M01 候选 → 随机抽样 → 存 crop + CSV。

用法：
    python tools/export_crop_review.py \
        --video ../data/whole_process_corner_1.mp4 --samples 10 \
        --video ../data/smoking_backward_1-1.mp4 --samples 12 \
        --out-dir results/crop_review/step2 --frames 900
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.action_recognizer import ActionRecognizer  # noqa: E402
from src.core.behavior_reasoner import BehaviorReasoner  # noqa: E402
from src.core.crop_classifier import CropClassifier  # noqa: E402
from src.core.detector import Detector  # noqa: E402
from src.core.pose_estimator import PoseEstimator  # noqa: E402
from src.core.tracker import Tracker  # noqa: E402
from src.pipeline.inference_engine import InferenceEngine  # noqa: E402


class _EngineShim:
    """复用引擎的 crop 提案/门控方法，但不初始化完整引擎。"""

    def __init__(self, classifier: CropClassifier) -> None:
        self.crop_classifier = classifier
        self._crop_review_cache: dict[str, tuple[int, str, float]] = {}
        self._crop_proposal_hits: dict[str, int] = {}

    _crop_verdict = InferenceEngine._crop_verdict
    _maybe_crop_proposal = InferenceEngine._maybe_crop_proposal
    _apply_crop_review = InferenceEngine._apply_crop_review


def build(cfg: dict):
    runtime = dict(cfg.get("runtime", {}))
    runtime["execution_provider"] = "CPUExecutionProvider"
    runtime["device"] = "cpu"
    det = Detector(dict(cfg.get("detector", {}), runtime=dict(runtime)))
    trk = Tracker(dict(cfg.get("tracker", {})))
    pose_cfg = dict(cfg.get("pose", {}))
    pose_cfg["onnx_path"] = "models/pose/vitpose_wholebody_s_int8.onnx"
    pose_cfg["runtime"] = dict(runtime)
    pose = PoseEstimator(pose_cfg)
    act_cfg = dict(cfg.get("action", {}))
    act_cfg["runtime"] = dict(runtime)
    act_cfg["active_onnx_path"] = act_cfg.get("onnx_path", "models/action/slowfast.onnx")
    ar = ActionRecognizer(act_cfg)
    clf = CropClassifier(dict(cfg.get("crop_classifier", {})))
    return det, trk, pose, ar, clf


def collect(video_path: Path, max_frames: int, det, trk, pose, ar, clf, cfg: dict) -> list[dict]:
    reasoner = BehaviorReasoner(dict(cfg.get("behavior_heuristics", {})))
    shim = _EngineShim(clf)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fid = 0
    records: list[dict] = []
    while fid < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        tracks = sorted(trk.update(det.infer(frame)), key=lambda t: float(t.get("score", 0.0)), reverse=True)[:3]
        if not tracks:
            continue
        for track in tracks:
            tid = str(track.get("track_id"))
            bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
            x, y, w, h = [int(v) for v in bbox]
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if not crop.size:
                continue
            kps = pose.infer(crop, {"bbox": [x, y, w, h]})
            ar._append_pose(tid, kps)
            pc_label, pc_conf = ar.infer_pose_classifier(tid)
            label, conf, source, _tags = reasoner.refine_light(
                track_id=tid, bbox=bbox, keypoints=kps, frame_shape=shape,
                base_label=pc_label, base_confidence=pc_conf, peer_tracks=tracks, frame_id=fid)
            label2, conf2, src2, proposed = shim._maybe_crop_proposal(
                track, frame, fid, label, conf, source, pc_label, pc_conf)
            label3, conf3, src3 = shim._apply_crop_review(
                track, frame, fid, label2, conf2, src2, pc_label, pc_conf)
            if label3 != "M01":
                continue
            pred = clf.predict(crop)
            records.append({
                "video": video_path.name,
                "frame_id": fid,
                "time_sec": round(fid / max(1e-6, fps), 2),
                "track_id": tid,
                "bbox": json.dumps([round(v, 1) for v in bbox]),
                "source": "crop_proposal" if proposed else "rule_agree",
                "pc_label": pc_label,
                "pc_conf": round(float(pc_conf), 3),
                "crop_pred": pred.get("label", ""),
                "crop_conf": round(float(pred.get("confidence", 0.0)), 3),
                "crop_path": "",
                "decision": "",
                "final_label": "",
                "note": "",
            })
        if fid % 25 == 0:
            reasoner.clear_stale_tracks({str(t.get("track_id")) for t in tracks})
    cap.release()
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, help="可重复；与 --samples 顺序对应")
    parser.add_argument("--samples", type=int, default=12, help="每个视频随机抽样数（<=0 表示全量）")
    parser.add_argument("--frames", type=int, default=900, help="每个视频处理帧数上限")
    parser.add_argument("--out-dir", default="results/crop_review/step2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    det, trk, pose, ar, clf = build(cfg)
    if not clf.loaded:
        print("[review] crop classifier not loaded:", clf.error)
        return 1

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows: list[dict] = []
    audio_like: list[dict] = []

    for raw_video in args.video:
        video_path = Path(raw_video)
        if not video_path.is_absolute():
            video_path = (ROOT / raw_video).resolve()
        if not video_path.exists():
            print(f"[review] missing video: {video_path}")
            continue
        records = collect(video_path, args.frames, det, trk, pose, ar, clf, cfg)
        total = len(records)
        picked = records if args.samples <= 0 else rng.sample(records, min(args.samples, total))
        stem = video_path.stem
        video_dir = out_dir / stem
        video_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for idx, row in enumerate(picked, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_id"]) - 1)
            ok, frame = cap.read()
            if not ok:
                continue
            bbox = json.loads(row["bbox"])
            x, y, w, h = [int(v) for v in bbox]
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if not crop.size:
                continue
            crop_path = video_dir / f"{idx:03d}_f{row['frame_id']}_{row['source']}.jpg"
            cv2.imwrite(str(crop_path), crop)
            row = dict(row)
            row["crop_path"] = str(crop_path.relative_to(ROOT))
            rows.append(row)
        cap.release()
        print(f"[review] {stem}: candidates={total} sampled={len(picked)}", flush=True)

    if not rows:
        print("[review] no candidates")
        return 1
    csv_path = out_dir / "review.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[review] wrote {len(rows)} rows -> {csv_path}")
    print("[review] 填写 decision=accept/reject；final_label 可写修正类别（如 Normal/M02）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
