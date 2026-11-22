from __future__ import annotations

"""告警视频切片导出（docs/06_GAPS_DESIGN.md §1）。

路径 A（已实现）：本地文件源 → ffmpeg remux（-c copy，毫秒级、零画质损失）；
路径 B（能力已备）：帧序列 → H.264 编码（encode_frames）；流源旁路录制未接线，
                当前返回 stream_source_not_recorded，不影响告警主链路。

约定：
- 异步单工作线程 + 有界队列，submit() 永不阻塞主循环；
- 切片完成后把 clip_path 回写事件 JSON（等待 write_event 落盘，最多 3s）；
- 所有失败只记 clip_error，不改变告警本身。
"""

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.pipeline.video_writer import _find_ffmpeg_with_libx264, open_video_writer


@dataclass
class ClipRequest:
    task_id: str
    frame_id: int
    event_filename: str      # 相对 results/ 的事件文件名
    source_url: str
    video_time_sec: float
    need_overlay: bool = False
    before_sec: float = 5.0
    after_sec: float = 5.0


def _find_ffmpeg() -> str | None:
    """通用 ffmpeg（remux 不需要 libx264）。"""
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    candidates = [env_bin, shutil.which("ffmpeg"), "/usr/bin/ffmpeg",
                  "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def probe_duration(path: str) -> float:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return 0.0
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    if not os.path.exists(ffprobe):
        ffprobe = shutil.which("ffprobe") or ""
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def is_local_file(url: str) -> bool:
    if not url or url.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return False
    return os.path.exists(url)


def export_clip_file(src: str, t0: float, duration: float, out: str,
                     precise: bool = False) -> tuple[bool, str]:
    """路径 A：从本地视频按时间窗导出片段。

    precise=False：ffmpeg -ss(输入侧) -c copy，快但切点可能提前 ≤1 GOP；
    precise=True ：重编码对齐（需要 libx264）。
    """
    if duration <= 0.2:
        return False, "duration_too_short"
    if precise:
        ffmpeg = _find_ffmpeg_with_libx264()
    else:
        ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg_not_found" if not precise else "ffmpeg_with_libx264_not_found"

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # -ss 放 -i 前（快速定位到关键帧）；-t 必须放 -i 后（输出时长），
    # 放 -i 前是"输入读取时长"，配合关键帧偏移会产出超长片段（实测 6s → 9s）。
    base = [ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{max(0.0, float(t0)):.3f}", "-i", src,
            "-t", f"{float(duration):.3f}"]
    if precise:
        cmd = base + ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    else:
        cmd = base + ["-c", "copy", "-avoid_negative_ts", "make_zero",
                      "-movflags", "+faststart", str(out_path)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as exc:
        return False, f"ffmpeg_exception: {exc}"
    if proc.returncode != 0:
        return False, f"ffmpeg_exit_{proc.returncode}: {proc.stderr.strip()[:200]}"
    if not out_path.exists() or out_path.stat().st_size < 1024:
        return False, "output_empty"

    # copy 模式受 GOP 约束：关键帧间隔大的视频会长出很多（实测 6s 窗口 → 9s 文件）。
    # 时长偏差超阈值时自动重编码精修，保证证据片段时长可信。
    if not precise:
        actual = probe_duration(str(out_path))
        if actual > 0 and abs(actual - float(duration)) > max(0.8, float(duration) * 0.25):
            return export_clip_file(src, t0, duration, out, precise=True)
    return True, str(out_path.resolve())


def encode_frames(frames: Iterable[np.ndarray], fps: float,
                  size: tuple[int, int], out: str) -> tuple[bool, str]:
    """路径 B 能力：帧序列 → H.264（流源旁路录制接线后可复用）。"""
    writer = open_video_writer(str(out), fps, size)
    if not writer.isOpened():
        return False, getattr(writer, "last_error", "writer_open_failed")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    error = getattr(writer, "last_error", "")
    if error:
        return False, error
    return True, str(Path(out).resolve())


class ClipExporter:
    """异步切片器：单工作线程 + 有界队列，失败不影响告警。"""

    def __init__(self, output_dir: str, cfg: dict | None = None) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.before_sec = float(cfg.get("before_seconds", 5.0))
        self.after_sec = float(cfg.get("after_seconds", 5.0))
        self.precise = bool(cfg.get("precise", True))
        self.max_queue = max(1, int(cfg.get("max_queue", 8)))
        self.keep_days = float(cfg.get("keep_days", 7.0))
        self._root = Path(output_dir)
        self.clip_dir = self._root / str(cfg.get("subdir", "clips"))
        self.clip_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue[ClipRequest] = queue.Queue(maxsize=self.max_queue)
        self._stop_event = threading.Event()
        self._task_id = ""
        self._lock = threading.Lock()
        self._last_cleanup = 0.0
        self._stats = {"submitted": 0, "done": 0, "failed": 0,
                       "dropped": 0, "last_error": "", "last_clip": ""}
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._loop, name="clip-exporter", daemon=True)
            self._thread.start()

    # ── 外部接口 ──────────────────────────────────────────────
    def begin_task(self, task_id: str) -> None:
        """任务隔离：清空待处理队列，旧任务残留请求在处理时被丢弃。"""
        self._task_id = str(task_id or "")
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def submit(self, req: ClipRequest) -> bool:
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait(req)
        except queue.Full:
            with self._lock:
                self._stats["dropped"] += 1
                self._stats["last_error"] = "queue_full"
            return False
        with self._lock:
            self._stats["submitted"] += 1
        return True

    def status(self) -> dict:
        with self._lock:
            stats = dict(self._stats)
        stats.update({
            "enabled": self.enabled,
            "queue_length": self._queue.qsize(),
            "max_queue": self.max_queue,
            "before_sec": self.before_sec,
            "after_sec": self.after_sec,
        })
        return stats

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── 工作线程 ──────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                req = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(req)
            except Exception as exc:  # 切片失败不抛出线程
                with self._lock:
                    self._stats["failed"] += 1
                    self._stats["last_error"] = f"{type(exc).__name__}: {exc}"

    def _process(self, req: ClipRequest) -> None:
        if req.task_id != self._task_id:
            return  # 上一任务残留
        if req.need_overlay:
            ok, info = False, "overlay_not_supported"
        elif is_local_file(req.source_url):
            total = probe_duration(req.source_url)
            t0 = max(0.0, float(req.video_time_sec) - self.before_sec)
            duration = self.before_sec + self.after_sec
            if total > 0:
                duration = min(duration, max(0.5, total - t0))
            out = self.clip_dir / f"{Path(req.event_filename).stem}.mp4"
            ok, info = export_clip_file(
                req.source_url, t0, duration, str(out), precise=self.precise)
        else:
            ok, info = False, "stream_source_not_recorded"

        if ok:
            clip_path = info
            with self._lock:
                self._stats["done"] += 1
                self._stats["last_clip"] = clip_path
            self._write_back(req.event_filename, clip_path=clip_path)
        else:
            with self._lock:
                self._stats["failed"] += 1
                self._stats["last_error"] = info
            self._write_back(req.event_filename, clip_error=info)
        self._cleanup_if_due()

    def _write_back(self, event_filename: str, clip_path: str = "",
                    clip_error: str = "") -> None:
        """回写事件 JSON：等待 write_event 落盘（最多 3s），再原子替换。"""
        path = self._root / event_filename
        for _ in range(30):
            if path.exists():
                break
            time.sleep(0.1)
        if not path.exists():
            return
        with self._lock:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                target = data.get("alert") if isinstance(data, dict) and isinstance(data.get("alert"), dict) else data
                if not isinstance(target, dict):
                    return
                if clip_path:
                    target["clip_path"] = clip_path
                if clip_error:
                    target["clip_error"] = clip_error
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(path)
            except Exception:
                return

    def _cleanup_if_due(self) -> None:
        """按 keep_days 清理旧切片（最多每 60s 一次，避免频繁 stat）。"""
        now = time.time()
        if self.keep_days <= 0 or now - self._last_cleanup < 60:
            return
        self._last_cleanup = now
        cutoff = now - self.keep_days * 86400
        try:
            for path in self.clip_dir.glob("*.mp4"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except Exception:
                    continue
        except Exception:
            pass
