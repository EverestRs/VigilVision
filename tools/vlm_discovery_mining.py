"""离线 VLM 发现器挖掘（第二期）。

流程：
  1) mine     密集采样视频 → VLM 发现（或 --dry-run 只导帧）→ JSONL
  2) export   把 JSONL 导出为人工复核 CSV（decision/final_label/notes 留空待填）
  3) dataset  读取复核 CSV，把 accept 的样本裁成 person crop，写入 data/crop_dataset/<label>/

用法示例：
  python tools/vlm_discovery_mining.py mine --video data/smoking_backward_1-1.mp4 --interval-sec 3
  python tools/vlm_discovery_mining.py mine --video data/xxx.mp4 --dry-run --out-dir results/discovery/dryrun
  python tools/vlm_discovery_mining.py export --jsonl results/discovery/mining.jsonl --csv results/discovery/mining_review.csv
  python tools/vlm_discovery_mining.py dataset --review-csv results/discovery/mining_review.csv --out-dir data/crop_dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.vlm_discovery import DISCOVERY_PROMPT, MINING_PROMPT  # noqa: E402


def _build_mining_images(mid_frame, prev_frame, detector, max_px: int = 960) -> list:
    """挖掘输入：全景（红框标目标）+ 最大人物裁剪 + 前一帧全景。"""
    images = []
    frame = mid_frame
    h, w = frame.shape[:2]
    scale = min(1.0, max_px / max(1.0, float(max(h, w))))
    pano = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else frame.copy()
    pano = pano.copy()
    images.append(pano)
    if detector is not None:
        dets = sorted(detector.infer(frame) or [], key=lambda d: float(d.get("score", 0.0)), reverse=True)
        # 最多 3 个人物裁剪：只取最大目标会漏掉背景里的吸烟者
        for det in dets[:3]:
            left, top, bw, bh = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
            x1, y1 = int(max(0, left)), int(max(0, top))
            x2, y2 = int(min(w, left + bw)), int(min(h, top + bh))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            cv2.rectangle(pano, (int(x1 * scale), int(y1 * scale)),
                          (int(x2 * scale), int(y2 * scale)), (0, 0, 255), 2)
            crop = frame[y1:y2, x1:x2]
            if not crop.size:
                continue
            ch, cw = crop.shape[:2]
            short = min(ch, cw)
            if short > 0 and short < 320:
                s = 320.0 / short
                crop = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_CUBIC)
            elif max(ch, cw) > 640:
                s = 640.0 / max(ch, cw)
                crop = cv2.resize(crop, (int(cw * s), int(ch * s)))
            images.append(crop)
    if prev_frame is not None:
        ph, pw = prev_frame.shape[:2]
        ps = min(1.0, 640.0 / max(1.0, float(max(ph, pw))))
        images.append(cv2.resize(prev_frame, (int(pw * ps), int(ph * ps))) if ps < 1.0 else prev_frame)
    return images[:4]


def _load_cfg(name: str) -> dict:
    path = ROOT / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _video_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.mp4")))
        elif any(ch in pattern for ch in "*?["):
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(p)
    seen, uniq = set(), []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _sample_frames(cap: cv2.VideoCapture, fps: float, t: float, spread: float = 0.4) -> list:
    frames = []
    for dt in (-spread, 0.0, spread):
        idx = int(max(0.0, t + dt) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    return frames


def cmd_mine(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    reviewer = None
    if not args.dry_run:
        cfg = _load_cfg("vlm_config.yaml")
        if args.no_vllm:
            cfg["use_vllm"] = False
        from src.core.vlm_reviewer import VLMReviewer
        reviewer = VLMReviewer(cfg)
        print(f"[mine] VLM enabled={reviewer.enabled} use_vllm={reviewer.use_vllm} url={reviewer.vllm_url}")

    detector = None
    if args.with_crops:
        from src.core.detector import Detector
        detector = Detector(dict(_load_cfg("model_config.yaml").get("detector", {})))
        print("[mine] person crops enabled")

    total = 0
    for video in _video_paths(args.video):
        if not video.exists():
            print(f"[skip] missing {video}")
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[skip] cannot open {video}")
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames_total / fps if frames_total else 0.0
        times = [i * args.interval_sec for i in range(int(duration // max(0.1, args.interval_sec)) + 1)]
        if args.max_samples_per_video > 0:
            times = times[: args.max_samples_per_video]
        print(f"[mine] {video.name} fps={fps:.1f} duration={duration:.1f}s samples={len(times)}")

        for t in times:
            if args.limit and total >= args.limit:
                break
            frames = _sample_frames(cap, fps, float(t), args.spread_sec)
            if len(frames) < 2:
                continue
            prompt = MINING_PROMPT if args.prompt == "mining" else DISCOVERY_PROMPT
            images = (_build_mining_images(frames[len(frames) // 2], frames[0], detector)
                      if args.with_crops else frames)
            total += 1
            record = {
                "video": str(video),
                "time": round(float(t), 2),
                "sampled_at": round(time.time(), 2),
            }
            if args.dry_run:
                frame_dir = out_dir / "frames" / video.stem
                frame_dir.mkdir(parents=True, exist_ok=True)
                name = frame_dir / f"{t:07.1f}.jpg"
                cv2.imwrite(str(name), frames[len(frames) // 2])
                record.update({"frame": str(name), "status": "dry_run"})
            else:
                result = reviewer.review_clip(images, track_id=f"{video.stem}@{t:.1f}", prompt=prompt)
                record.update({
                    "label": str(result.get("label", "Normal")),
                    "confidence": round(float(result.get("confidence", 0.0)), 3),
                    "description": str(result.get("description", ""))[:300],
                    "latency_ms": int(result.get("latency_ms", 0) or 0),
                    "status": str(result.get("status", "error")),
                })
                print(f"  t={t:7.1f} -> {record['label']}({record['confidence']}) {record['status']}")
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        cap.release()
        if args.limit and total >= args.limit:
            break

    print(f"[mine] done: {total} samples -> {jsonl_path}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    rows = []
    with Path(args.jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "video", "time", "vlm_label", "vlm_conf", "description",
                         "decision", "final_label", "notes"])
        for i, row in enumerate(rows):
            writer.writerow([
                i, row.get("video", ""), row.get("time", ""),
                row.get("label", ""), row.get("confidence", ""), row.get("description", ""),
                "", "", "",
            ])
    print(f"[export] {len(rows)} rows -> {csv_path}（decision 填 accept/reject，final_label 填修正后的类别）")
    return 0


def _crop_largest_person(detector, frame):
    # detector.infer 返回 [{bbox: [left, top, w, h], score: float}]，解码阶段已只保留 person
    dets = [d for d in (detector.infer(frame) or []) if isinstance(d, dict) and d.get("bbox")]
    if not dets:
        return None
    best = max(dets, key=lambda d: float(d.get("score", 0.0)))
    left, top, bw, bh = [float(v) for v in best["bbox"]]
    x1, y1, x2, y2 = left, top, left + bw, top + bh
    h, w = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pw * 0.1)); y1 = int(max(0, y1 - ph * 0.1))
    x2 = int(min(w, x2 + pw * 0.1)); y2 = int(min(h, y2 + ph * 0.1))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def cmd_dataset(args: argparse.Namespace) -> int:
    from src.core.detector import Detector
    model_cfg = _load_cfg("model_config.yaml")
    det_cfg = dict(model_cfg.get("detector", {}))
    det_cfg.setdefault("runtime", {}).setdefault("execution_provider", "CPU")
    detector = Detector(det_cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    skipped = 0
    with Path(args.review_csv).open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if str(row.get("decision", "")).strip().lower() not in {"accept", "a", "1", "yes"}:
            continue
        label = str(row.get("final_label", "")).strip() or str(row.get("vlm_label", "")).strip()
        if not label:
            skipped += 1
            continue
        if label == "Normal" and not args.include_normal:
            continue
        video = Path(row.get("video", ""))
        if not video.exists():
            skipped += 1
            continue
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        t = float(row.get("time", 0.0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            skipped += 1
            continue
        crop = _crop_largest_person(detector, frame)
        if crop is None:
            skipped += 1
            continue
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        name = f"{video.stem}__{t:07.1f}.jpg"
        cv2.imwrite(str(label_dir / name), crop)
        counts[label] = counts.get(label, 0) + 1
    print(f"[dataset] written={sum(counts.values())} skipped={skipped} dist={counts} -> {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="离线 VLM 发现器挖掘")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mine = sub.add_parser("mine")
    p_mine.add_argument("--video", action="append", required=True, help="视频路径/目录/glob，可重复")
    p_mine.add_argument("--interval-sec", type=float, default=2.0,
                        help="采样间隔；跌倒也常只有 1~2s，建议 1~2s")
    p_mine.add_argument("--prompt", choices=["mining", "discovery"], default="mining",
                        help="mining=逐人排查（离线）；discovery=运行时保守 prompt")
    p_mine.add_argument("--with-crops", action="store_true",
                        help="附加最大人物裁剪（推荐：远距/小目标必需）")
    p_mine.add_argument("--spread-sec", type=float, default=0.4, help="每个采样点前后各取一帧的时间差")
    p_mine.add_argument("--max-samples-per-video", type=int, default=0)
    p_mine.add_argument("--limit", type=int, default=0, help="全局调用上限（控制 VLM 成本）")
    p_mine.add_argument("--dry-run", action="store_true", help="只导帧与 manifest，不调用 VLM")
    p_mine.add_argument("--no-vllm", action="store_true", help="不用 vLLM 服务，改用 transformers 本地加载")
    p_mine.add_argument("--jsonl", default="results/discovery/mining.jsonl")
    p_mine.add_argument("--out-dir", default="results/discovery/frames")
    p_mine.set_defaults(func=cmd_mine)

    p_export = sub.add_parser("export")
    p_export.add_argument("--jsonl", required=True)
    p_export.add_argument("--csv", required=True)
    p_export.set_defaults(func=cmd_export)

    p_ds = sub.add_parser("dataset")
    p_ds.add_argument("--review-csv", required=True)
    p_ds.add_argument("--out-dir", default="data/crop_dataset")
    p_ds.add_argument("--include-normal", action="store_true", help="把 accept 的 Normal 也作为负样本写入")
    p_ds.set_defaults(func=cmd_dataset)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
