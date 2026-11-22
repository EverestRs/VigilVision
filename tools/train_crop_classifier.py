"""2D crop 分类器训练（第三期 triage 模型）。

数据来自 tools/vlm_discovery_mining.py dataset 产出的目录结构：
    data/crop_dataset/<label>/<video>__<time>.jpg

切分按"视频"分组（文件名 __ 前缀），避免同源帧泄漏到验证集。
输出：
    models/crop_classifier/best.pth
    models/crop_classifier/labels.json
    models/crop_classifier/metrics.json
    models/crop_classifier/crop_classifier.onnx（--export-onnx）
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]


class CropDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], img_size: int, train: bool) -> None:
        self.items = items
        self.img_size = img_size
        self.train = train
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5) if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.2, 0.2, 0.1) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.tf(image), label


def _group_of(path: Path) -> str:
    return path.stem.split("__")[0]


def collect(data_dir: Path) -> tuple[list[tuple[Path, int]], list[str], dict[str, int]]:
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if not classes:
        raise SystemExit(f"no class folders under {data_dir}")
    items: list[tuple[Path, int]] = []
    dist: dict[str, int] = {}
    for idx, name in enumerate(classes):
        files = sorted((data_dir / name).glob("*.jpg")) + sorted((data_dir / name).glob("*.png"))
        dist[name] = len(files)
        items.extend((p, idx) for p in files)
    return items, classes, dist


def split_by_group(items, val_ratio: float, seed: int, classes: list[str]):
    rng = random.Random(seed)
    groups = sorted({_group_of(p) for p, _ in items})
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_ratio)) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val])
    train = [(p, y) for p, y in items if _group_of(p) not in val_groups]
    val = [(p, y) for p, y in items if _group_of(p) in val_groups]
    if not train or not val:  # 单视频/样本太少时退化为随机切分
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_ratio)))
        train, val = items[:cut], items[cut:] or items[:1]
    return train, val


def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += int((preds == labels).sum())
            total += int(labels.numel())
            for y, p in zip(labels.tolist(), preds.tolist()):
                per_class_total[y] += 1
                per_class_correct[y] += int(y == p)
    recall = {str(i): round(per_class_correct[i] / max(1, per_class_total[i]), 4)
              for i in range(num_classes)}
    return (correct / max(1, total)), recall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/crop_dataset")
    parser.add_argument("--out-dir", default="models/crop_classifier")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    items, classes, dist = collect(Path(args.data_dir))
    print(f"classes={classes} dist={dist}")
    train_items, val_items = split_by_group(items, args.val_ratio, args.seed, classes)
    print(f"train={len(train_items)} val={len(val_items)}")

    train_loader = DataLoader(CropDataset(train_items, args.img_size, True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val_items, args.img_size, False),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = None
    counts = np.array([dist[c] for c in classes], dtype=np.float32)
    if (counts > 0).all():
        class_w = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))).astype(np.float32))
        weights = class_w.to(args.device)

    model = mobilenet_v3_small(weights="DEFAULT" if args.pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(classes))
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")

    best_acc = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
        scheduler.step()
        acc, recall = evaluate(model, val_loader, args.device, len(classes))
        epoch_loss = running / max(1, len(train_items))
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss, 4), "val_acc": round(acc, 4)})
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, out_dir / "best.pth")
            (out_dir / "metrics.json").write_text(json.dumps({
                "best_val_acc": round(best_acc, 4),
                "epochs": args.epochs,
                "total_samples": len(items),
                "per_class": {c: dist[c] for c in classes},
                "val_recall": recall,
                "history": history,
            }, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.export_onnx:
        model.load_state_dict(torch.load(out_dir / "best.pth", map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        dummy = torch.zeros(1, 3, args.img_size, args.img_size)
        torch.onnx.export(model, dummy, str(out_dir / "crop_classifier.onnx"),
                          input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
        print("exported:", out_dir / "crop_classifier.onnx")
    print("done. best_val_acc=", round(best_acc, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
