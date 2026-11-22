from __future__ import annotations

"""视频写盘后端：优先 ffmpeg（H.264 / yuv420p / +faststart），不可用时回退 OpenCV。

设计说明见 docs/06_GAPS_DESIGN.md §2：
- 逐帧算法在 numpy/cv2 侧完成，编码/封装交给 ffmpeg；
- 统一输出浏览器可播格式，修复 render 任务 mp4v 兼容性问题；
- 接口对齐 cv2.VideoWriter（isOpened/write/release），调用方改动最小。
"""

import os
import shutil
import subprocess

import cv2
import numpy as np

# ffmpeg 能力探测结果（模块级缓存，避免每个 writer 重复 spawn）
_FFMPEG_X264_BIN: str | None = None
_FFMPEG_PROBED = False


def _find_ffmpeg_with_libx264() -> str | None:
    """找一个支持 libx264 的 ffmpeg。

    不能直接用 shutil.which：conda/venv 里可能装的是无 libx264 的精简版
    （实测 VdAnalyst 环境的 ffmpeg 仅 432KB，`-encoders` 无 libx264，
    启动后立即退出 → 管道 BrokenPipe）。按 FFMPEG_BIN → PATH → 常见路径逐个探测。
    """
    global _FFMPEG_X264_BIN, _FFMPEG_PROBED
    if _FFMPEG_PROBED:
        return _FFMPEG_X264_BIN

    candidates: list[str] = []
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        candidates.append(env_bin)
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)
    candidates.extend(["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not os.path.exists(candidate):
            continue
        seen.add(candidate)
        try:
            out = subprocess.run(
                [candidate, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except Exception:
            continue
        if "libx264" in out:
            _FFMPEG_X264_BIN = candidate
            _FFMPEG_PROBED = True
            return candidate

    _FFMPEG_PROBED = True
    _FFMPEG_X264_BIN = None
    return None


class FfmpegVideoWriter:
    """rawvideo(bgr24) → libx264 / yuv420p / +faststart。"""

    codec = "h264"

    def __init__(self, path: str, fps: float, size: tuple[int, int],
                 crf: int = 23, preset: str = "veryfast") -> None:
        self.path = str(path)
        self.fps = float(fps) if fps and float(fps) > 1.0 else 25.0
        self.size = (int(size[0]), int(size[1]))
        self.last_error = ""
        self._proc: subprocess.Popen | None = None

        ffmpeg = _find_ffmpeg_with_libx264()
        if not ffmpeg:
            self.last_error = "ffmpeg_with_libx264_not_found"
            return

        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.size[0]}x{self.size[1]}",
            "-r", f"{self.fps:.6f}",
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264", "-preset", str(preset), "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self.path,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # 进程创建失败 → 交由工厂回退
            self.last_error = f"ffmpeg_spawn_failed: {exc}"
            self._proc = None

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            return
        try:
            height, width = frame.shape[:2]
            if (width, height) != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            # ffmpeg 提前退出（参数错误/磁盘满）：抛给调用方标 failed，
            # 避免"任务 finished 但文件坏"的假成功。
            self.last_error = f"ffmpeg_write_failed: {exc}"
            try:
                self._proc.kill()
            except Exception:
                pass
            raise RuntimeError(self.last_error) from exc

    def release(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if proc.returncode not in (0, None) and not self.last_error:
            self.last_error = f"ffmpeg_exit_{proc.returncode}"


class CvVideoWriter:
    """OpenCV 回退（mp4v）：零外部依赖，但浏览器兼容性差。"""

    codec = "mp4v"

    def __init__(self, path: str, fps: float, size: tuple[int, int]) -> None:
        fps = float(fps) if fps and float(fps) > 1.0 else 25.0
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(size[0]), int(size[1]))
        )
        self.last_error = "" if self._writer.isOpened() else "cv_writer_open_failed"

    def isOpened(self) -> bool:
        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


def open_video_writer(path: str, fps: float, size: tuple[int, int],
                      backend: str = "auto") -> FfmpegVideoWriter | CvVideoWriter:
    """工厂：auto 优先 ffmpeg，失败回退 cv2；backend=cv 可强制旧行为（回滚开关）。"""
    if backend in ("auto", "ffmpeg"):
        writer = FfmpegVideoWriter(path, fps, size)
        if writer.isOpened():
            return writer
        if backend == "ffmpeg":
            return writer
    return CvVideoWriter(path, fps, size)
