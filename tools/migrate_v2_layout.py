from __future__ import annotations

"""一次性迁移：把 training_data_v2 规范成与 v1 兼容的布局。

背景（docs/05 §4.1、docs/06）：
- v1（results/training_data）是 07-24 旧口径，但它的**布局**被 FastPoseDataset
  与 5 个工具硬编码依赖：manifest 字段为 int/float、npz 平铺在根目录；
- v2（results/training_data_v2）是当前生产口径，但 manifest 字段是 str、
  npz 在 npz/ 子目录——直接切默认路径会 FileNotFoundError / TypeError。

本脚本只做两件事（幂等）：
1. manifest.json 字段类型转换：frame_id/class_idx → int，refined_confidence → float；
2. npz/*.npz → 根目录平铺。
原 manifest 备份为 manifest_raw.json（仅首次）。

用法:
  python tools/migrate_v2_layout.py --dry-run
  python tools/migrate_v2_layout.py
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="results/training_data_v2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: manifest 不存在 {manifest_path}")
        return 1

    samples = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        print("FAIL: manifest 不是非空 list")
        return 1

    # 1) 字段类型规范化
    changed = 0
    for sample in samples:
        dirty = False
        for key, cast in (("frame_id", int), ("class_idx", int),
                          ("refined_confidence", float)):
            if key in sample and not isinstance(sample[key], cast):
                try:
                    sample[key] = cast(str(sample[key]).strip())
                    dirty = True
                except (TypeError, ValueError):
                    pass
        if dirty:
            changed += 1

    # 2) npz 平铺
    npz_dir = root / "npz"
    moved = 0
    skipped = 0
    if npz_dir.exists():
        for src in sorted(npz_dir.glob("*.npz")):
            dst = root / src.name
            if dst.exists():
                skipped += 1
                continue
            if not args.dry_run:
                shutil.move(str(src), str(dst))
            moved += 1

    # 3) 校验每个 video 有对应 npz
    missing = []
    for video in sorted({str(s.get("video", "")) for s in samples}):
        stem = Path(video).stem
        if not stem:
            continue
        if not (root / f"{stem}.npz").exists() and not (npz_dir / f"{stem}.npz").exists():
            missing.append(stem)

    print(f"[migrate] data_dir={root} samples={len(samples)} "
          f"converted={changed} moved_npz={moved} skipped={skipped} missing={len(missing)}")
    if missing:
        print("  MISSING npz:", ", ".join(missing[:10]))

    if args.dry_run:
        print("[migrate] dry-run，未写盘")
        return 0

    # 4) 备份原 manifest（仅首次）并写回
    raw_backup = root / "manifest_raw.json"
    if not raw_backup.exists():
        shutil.copy2(manifest_path, raw_backup)
    manifest_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")

    # 5) 空 npz 目录清理
    if npz_dir.exists() and not any(npz_dir.iterdir()):
        npz_dir.rmdir()

    print("[migrate] 完成：manifest 已规范化，npz 已平铺")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
