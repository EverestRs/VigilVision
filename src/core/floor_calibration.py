"""自动地面标定核心：脚点统计 → 最低簇 → 凸包 → 与安全带并集。

设计目标：**不会比现有地面线更差**。任何自动拟合结果都会与
`[0, floor_line] → [1, floor_line] → [1, 1] → [0, 1]` 的底部安全带取并集；
点数不足/面积异常时直接退化为安全带。供 CLI 与 REST 后台任务共用。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ProgressCb = Callable[[int, int], None]


def _open_capture(source: str, is_stream: bool, open_timeout_ms: int = 5000,
                  read_timeout_ms: int = 5000) -> cv2.VideoCapture:
    if is_stream:
        cap = cv2.VideoCapture(source)
        # OpenCV >= 4.5 支持超时；不支持时 set 返回 False，不影响使用
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_ms))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_ms))
        return cap
    return cv2.VideoCapture(str(source))


def collect_points(video: Path | str, det, pose, reasoner, sample_sec: float, max_frames: int,
                   progress_cb: ProgressCb | None = None, *,
                   is_stream: bool = False, duration_sec: float | None = None,
                   open_timeout_ms: int = 5000,
                   read_timeout_ms: int = 5000) -> tuple[list[tuple[float, float]], int]:
    """采样地面接触点。

    - 文件模式：按 CAP_PROP_POS_FRAMES 跳帧（快）
    - 流模式（RTSP/HTTP）：顺序读、按帧号取模采样，支持 duration_sec 到时停止；
      不 seek，避免 RTSP 每次 seek 造成数秒延迟。
    """
    source = str(video)
    cap = _open_capture(source, is_stream, open_timeout_ms, read_timeout_ms)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开{'视频流' if is_stream else '视频文件'}: {source}")
    if is_stream:
        return _collect_stream(cap, det, pose, sample_sec, max_frames, progress_cb, duration_sec)

    video_path = Path(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(sample_sec * fps)))
    points: list[tuple[float, float]] = []
    frames_used = 0
    frame_id = 0
    while frames_used < max_frames and frame_id < (total or 10 ** 9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, img = cap.read()
        if not ok:
            break
        frame_id += step
        frames_used += 1
        points.extend(_points_from_frame(img, det, pose))
        if progress_cb is not None:
            progress_cb(frames_used, max_frames)
    cap.release()
    return points, frames_used


def _points_from_frame(img: np.ndarray, det, pose) -> list[tuple[float, float]]:
    """单帧地面接触点：踝点（下半区）+ 人体框底边中心。"""
    h, w = img.shape[:2]
    out: list[tuple[float, float]] = []
    dets = sorted(det.infer(img) or [], key=lambda d: float(d.get("score", 0.0)), reverse=True)[:4]
    for d in dets:
        bbox = [float(v) for v in d.get("bbox", [0, 0, 0, 0])]
        x, y, bw, bh = [int(v) for v in bbox]
        crop = img[max(0, y):y + bh, max(0, x):x + bw]
        if crop.size == 0:
            continue
        kps = pose.infer(crop, {"bbox": [x, y, bw, bh]})
        for idx in (15, 16):
            if idx < len(kps):
                kx, ky, kc = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
                if kc >= 0.40 and 0 <= kx + x < w and 0.45 * h <= ky + y < h:
                    out.append(((kx + x) / w, (ky + y) / h))
        # 框底边中心也作为接触点：躺地/蹲坐的人同样提供地面证据
        fx, fy = (x + bw / 2) / w, (y + bh) / h
        if fy >= 0.45:
            out.append((fx, fy))
    return out


def _collect_stream(cap: cv2.VideoCapture, det, pose, sample_sec: float, max_frames: int,
                    progress_cb: ProgressCb | None,
                    duration_sec: float | None) -> tuple[list[tuple[float, float]], int]:
    """流模式：顺序读取，不 seek；按 duration / max_frames 停止。"""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(sample_sec * fps)))
    points: list[tuple[float, float]] = []
    frames_used = 0
    read = 0
    started = time.monotonic()
    while frames_used < max_frames:
        if duration_sec and duration_sec > 0 and time.monotonic() - started >= duration_sec:
            break
        ok, img = cap.read()
        if not ok:
            break
        read += 1
        if read % step != 0:
            continue
        points.extend(_points_from_frame(img, det, pose))
        frames_used += 1
        if progress_cb is not None:
            progress_cb(frames_used, max_frames)
    cap.release()
    if frames_used == 0:
        raise RuntimeError("视频流未读到任何帧（检查地址/网络/编码）")
    return points, frames_used


def select_lower_cluster(points: list[tuple[float, float]], min_keep: int = 20):
    """对脚点 y 做 1D 2-means，保留更靠下的簇，剔除桌面/遮挡误点。"""
    if len(points) < min_keep * 2:
        return points
    ys = np.array([p[1] for p in points], dtype=np.float32)
    c1, c2 = np.percentile(ys, 25), np.percentile(ys, 75)
    for _ in range(12):
        m1, m2 = np.abs(ys - c1) <= np.abs(ys - c2), np.abs(ys - c1) > np.abs(ys - c2)
        if not m1.any() or not m2.any():
            break
        c1, c2 = float(ys[m1].mean()), float(ys[m2].mean())
    threshold = (c1 + c2) / 2.0
    kept = [p for p in points if p[1] >= threshold]
    return kept if len(kept) >= min_keep else points


def build_polygon(points: list[tuple[float, float]], margin: float = 0.03) -> list[list[float]]:
    arr = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(arr.reshape(-1, 1, 2)).reshape(-1, 2)
    center = hull.mean(axis=0)
    hull = (hull - center) * (1.0 + margin * 2.0) + center
    hull[:, 0] = np.clip(hull[:, 0], 0.0, 1.0)
    hull[:, 1] = np.clip(hull[:, 1], 0.0, 1.0)
    if len(hull) > 12:
        simplified = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.01, True).reshape(-1, 2)
        if len(simplified) >= 3:
            hull = simplified
    return [[round(float(x), 4), round(float(y), 4)] for x, y in hull]


def polygon_area(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    arr = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(arr)))


def fuse_with_fallback_band(polygon: list[list[float]], floor_line: float) -> list[list[float]]:
    pts = [(float(x), float(y)) for x, y in polygon]
    pts += [(0.0, floor_line), (1.0, floor_line), (1.0, 1.0), (0.0, 1.0)]
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr.reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) > 12:
        simplified = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.01, True).reshape(-1, 2)
        if len(simplified) >= 3:
            hull = simplified
    return [[round(float(np.clip(x, 0, 1)), 4), round(float(np.clip(y, 0, 1)), 4)]
            for x, y in hull]


def calibrate_from_video(video: Path | str, det, pose, reasoner, *, floor_line: float = 0.90,
                         sample_sec: float = 1.0, max_frames: int = 120,
                         min_points: int = 30,
                         progress_cb: ProgressCb | None = None,
                         is_stream: bool = False,
                         duration_sec: float | None = None) -> dict[str, Any]:
    raw_points, frames_used = collect_points(
        video, det, pose, reasoner, sample_sec, max_frames, progress_cb,
        is_stream=is_stream, duration_sec=duration_sec,
    )
    points = select_lower_cluster(raw_points)
    ys = [p[1] for p in points]
    floor_line = float(min(0.99, max(0.40, floor_line)))
    summary: dict[str, Any] = {
        "video": str(video),
        "source_mode": "stream" if is_stream else "file",
        "duration_sec": duration_sec,
        "frames_used": frames_used,
        "raw_foot_points": len(raw_points),
        "foot_points": len(points),
        "y_min": round(min(ys), 3) if ys else None,
        "y_p10": round(float(np.percentile(ys, 10)), 3) if ys else None,
        "y_median": round(float(np.percentile(ys, 50)), 3) if ys else None,
        "floor_line_used": floor_line,
    }
    band = [[0.0, floor_line], [1.0, floor_line], [1.0, 1.0], [0.0, 1.0]]
    if len(points) < min_points:
        summary.update({"result": "insufficient_points_fallback_band", "polygon": band,
                        "polygon_points": 4})
        return summary

    auto_polygon = build_polygon(points)
    area = polygon_area(auto_polygon)
    summary.update({"auto_polygon": auto_polygon, "auto_area": round(area, 4)})
    if area < 0.01 or area > 0.85:
        summary.update({"result": "auto_rejected_fallback_band", "polygon": band,
                        "polygon_points": 4})
    else:
        fused = fuse_with_fallback_band(auto_polygon, floor_line)
        summary.update({"result": "auto_fused_with_fallback_band", "polygon": fused,
                        "polygon_points": len(fused)})
    summary["suggested_floor_line_ratio"] = round(
        max(0.4, float(np.percentile(ys, 10)) - 0.02), 3) if ys else floor_line
    return summary
