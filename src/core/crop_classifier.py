"""2D crop 分类器：M01/M02/H03 的本地确认钥匙。

背景：PoseClassifier 的标签来自弱规则，灰区又依赖 VLM（慢、且与检测抢 GPU）；
已有的 7 类 crop 模型（mobilenet_v3_small，验证集 92.4%，M01 阈值 0.4 时 P=0.805/R=0.994）
可以在本地做"确认/否决"，把明显不是吸烟的坐姿帧直接按掉，减少 VLM 调用与误报。

设计原则：
- 默认跑 CPU（mobilenet 小模型 ~10ms），避免与 ONNX 检测抢 GPU；
- 只在配置的类别上生效（默认 M01/M02/H03），按 track 节流缓存；
- `confirm_alerts=false`（默认）时只做**否决**；设为 true 才把本地确认当作告警钥匙。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - 环境缺少 onnxruntime 时降级
    ort = None

DEFAULT_CLASSES = ["Normal", "M01", "M02", "H03", "Wave", "Throw", "Handshake"]
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class CropClassifier:
    """ONNX 推理封装（懒加载，失败不影响主链路）。"""

    def __init__(self, config: dict[str, Any] | None, root: Path | str | None = None) -> None:
        cfg = config or {}
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.enabled = bool(cfg.get("enabled", False))
        self.input_size = int(cfg.get("input_size", 224))
        self.review_classes = {str(c) for c in (cfg.get("classes") or ["M01", "M02", "H03"])}
        self.throttle_frames = int(max(1, cfg.get("throttle_frames", 30)))
        self.device = str(cfg.get("device", "cpu"))
        self.confirm_alerts = bool(cfg.get("confirm_alerts", False))
        # M01 提案通道：crop 作为主提案器，pose 近嘴规则降级为佐证
        self.propose_m01 = bool(cfg.get("propose_m01", True))
        self.proposal_conf = float(cfg.get("proposal_conf", 0.70))
        self.proposal_hits = int(max(1, cfg.get("proposal_hits", 2)))
        self.m01_require_crop = bool(cfg.get("m01_require_crop", True))
        self.onnx_path = self._path(cfg.get("onnx_path", "models/crop_classifier/crop_classifier.onnx"))
        self.labels_path = self._path(cfg.get("labels_path", "models/crop_classifier/labels.json"))
        self.thresholds_path = self._path(cfg.get("thresholds_path", "models/crop_classifier/thresholds.json"))

        self.session = None
        self.classes: list[str] = list(DEFAULT_CLASSES)
        self.class_thresholds: dict[str, float] = {}
        self.normal_conf = 0.85
        self.loaded = False
        self.error = ""
        if self.enabled:
            self._load()

    def _path(self, value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def _load(self) -> None:
        try:
            if ort is None:
                raise RuntimeError("onnxruntime not installed")
            if not self.onnx_path.exists():
                raise FileNotFoundError(str(self.onnx_path))
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if self.device == "cuda" else ["CPUExecutionProvider"])
            self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            if self.labels_path.exists():
                self.classes = list(json.loads(self.labels_path.read_text(encoding="utf-8")))
            if self.thresholds_path.exists():
                data = json.loads(self.thresholds_path.read_text(encoding="utf-8"))
                self.class_thresholds = {
                    str(k): float(v) for k, v in (data.get("class_thresholds") or {}).items()
                }
                self.normal_conf = float(data.get("normal_conf", self.normal_conf))
            self.loaded = True
        except Exception as exc:
            self.error = str(exc)[:160]
            self.loaded = False

    def predict(self, crop: np.ndarray | None) -> dict[str, Any]:
        if not self.loaded or crop is None or getattr(crop, "size", 0) == 0:
            return {"label": "", "confidence": 0.0}
        try:
            image = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
            logits = np.asarray(self.session.run(None, {"image": tensor})[0][0], dtype=np.float32)
            probs = np.exp(logits - np.max(logits))
            probs /= max(1e-6, float(probs.sum()))
            index = int(np.argmax(probs))
            label = self.classes[index] if index < len(self.classes) else ""
            return {"label": label, "confidence": float(probs[index])}
        except Exception as exc:
            self.error = f"predict_failed: {exc}"[:160]
            return {"label": "", "confidence": 0.0}

    def threshold_for(self, label: str) -> float:
        return float(self.class_thresholds.get(str(label), 0.5))

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "error": self.error,
            "device": self.device,
            "classes": list(self.classes),
            "review_classes": sorted(self.review_classes),
            "class_thresholds": dict(self.class_thresholds),
            "normal_conf": self.normal_conf,
            "confirm_alerts": self.confirm_alerts,
            "propose_m01": self.propose_m01,
            "proposal_conf": self.proposal_conf,
            "proposal_hits": self.proposal_hits,
            "m01_require_crop": self.m01_require_crop,
        }


def decide_crop_action(
    label: str,
    confidence: float,
    src_tag: str,
    pred_label: str,
    pred_conf: float,
    *,
    review_classes: set[str],
    thresholds: dict[str, float],
    normal_conf: float,
    confirm_alerts: bool,
    base_label: str,
    base_confidence: float,
) -> tuple[str, float, str, str]:
    """根据本地分类器结论决定是否确认/否决。

    返回 (label, confidence, src_tag, action)：
      - confirm: 同类且过阈值；confirm_alerts=True 时 src 标记 crop_confirmed（可作告警钥匙）
      - veto:    判为 Normal 且置信度足够 → 回退 base，src 标记 crop_veto
      - abstain/skip: 保持原样
    """
    if label not in review_classes or not pred_label:
        return label, confidence, src_tag, "skip"
    if pred_label == label and pred_conf >= float(thresholds.get(label, 0.5)):
        if confirm_alerts:
            return label, max(float(confidence), float(pred_conf)), "crop_confirmed", "confirm"
        return label, confidence, src_tag, "confirm_display"
    if pred_label == "Normal" and pred_conf >= float(normal_conf):
        return base_label, base_confidence, "crop_veto", "veto"
    return label, confidence, src_tag, "abstain"
