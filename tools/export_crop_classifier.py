"""把 results/crop_dataset/best.pth 导出为 ONNX，并写 labels/thresholds 与验证指标。

用法：
    python tools/export_crop_classifier.py \
        --ckpt results/crop_dataset/best.pth \
        --out-dir models/crop_classifier

输出：
    models/crop_classifier/crop_classifier.onnx
    models/crop_classifier/labels.json
    models/crop_classifier/thresholds.json
    models/crop_classifier/metrics.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]


def build_model(num_classes: int) -> torch.nn.Module:
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def preprocess(image: np.ndarray, img_size: int) -> np.ndarray:
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return ((rgb - mean) / std).transpose(2, 0, 1)[None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="results/crop_dataset/best.pth")
    parser.add_argument("--out-dir", default="models/crop_classifier")
    parser.add_argument("--manifest", default="results/crop_dataset/manifest_val.jsonl")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    ckpt_path = ROOT / args.ckpt
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    classes = list(ckpt["classes"])
    model = build_model(len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} classes={classes} val_acc={ckpt.get('val_acc')}")

    onnx_path = out_dir / "crop_classifier.onnx"
    dummy = torch.zeros(1, 3, args.img_size, args.img_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("exported:", onnx_path, onnx_path.stat().st_size)

    (out_dir / "labels.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")

    # 阈值：沿用离线调优结果；Normal 翻案阈值取保守值
    tuned_path = ROOT / "results" / "crop_dataset" / "thresholds_tuned.json"
    class_thresholds: dict[str, float] = {}
    if tuned_path.exists():
        tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
        class_thresholds = {k: float(v) for k, v in (tuned.get("class_thresholds") or {}).items()}
    (out_dir / "thresholds.json").write_text(json.dumps({
        "class_thresholds": class_thresholds,
        "normal_conf": 0.85,
        "note": "class_thresholds 来自 thresholds_tuned.json；normal_conf 是 Normal 否决所需置信度",
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ONNX 验证（与训练脚本一致的预处理）
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    manifest = ROOT / args.manifest
    if not manifest.exists():
        print("manifest missing, skip validation")
        return 0
    correct = total = 0
    per_class_ok = Counter()
    per_class_total = Counter()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = ROOT / "results" / "crop_dataset" / row["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        logits = session.run(None, {"image": preprocess(image, args.img_size)})[0][0]
        pred = classes[int(np.argmax(logits))]
        total += 1
        per_class_total[row["cls"]] += 1
        per_class_ok[row["cls"]] += int(pred == row["cls"])
        correct += int(pred == row["cls"])
    acc = correct / max(1, total)
    recall = {c: round(per_class_ok[c] / max(1, per_class_total[c]), 4) for c in classes}
    metrics = {"val_acc": round(acc, 4), "val_samples": total, "per_class_recall": recall}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print("onnx val:", json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
