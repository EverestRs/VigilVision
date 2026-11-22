from __future__ import annotations

"""视频流读取三档后端（T2.2，01 §2.2）：cv 软解 / ffmpeg 管道 / auto 自动降级。

- 目标是"解码 CPU 预算"而非绑定 VDEC（G2 精化）：A=cv（子码流）→ B=ffmpeg 管道；
  auto 在 cv 打不开或持续读失败时自动切 ffmpeg（盒上由 g2 探针实测选档后可显式指定）；
- read() -> (ok, frame, frame_id) 契约与断线重连语义与旧版完全一致；
- ffmpeg 后端经 rawvideo/bgr24 管道出帧，分辨率固定（管道按字节对齐读帧）。
"""

import shutil
import subprocess
import time
from typing import Any

import cv2
import numpy as np


class _CvBackend:
    name = "cv"

    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)
        if url.startswith("rtsp://"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def opened(self) -> bool:
        return bool(self.cap.isOpened())

    def fps(self) -> float:
        try:
            v = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        except Exception:
            v = 0.0
        return v if v > 1.0 else 0.0

    def read(self):
        return self.cap.read()

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


class _FfmpegBackend:
    name = "ffmpeg"

    def __init__(self, url: str, width: int = 1280, height: int = 720):
        self.url = url
        self.width, self.height = int(width), int(height)
        self._frame_bytes = self.width * self.height * 3
        self._proc = None
        self._start()

    def _cmd(self) -> list[str]:
        ffmpeg = shutil.which("ffmpeg")
        base = [ffmpeg or "ffmpeg", "-loglevel", "error"]
        url = self.url
        is_device = url.isdigit()
        if url.startswith("rtsp://"):
            base += ["-rtsp_transport", "tcp"]
        elif not url.startswith(("http", "/")) and not is_device:
            base += ["-f", "lavfi"]  # 合成测试源（testsrc=…），与 g2 探针口径一致
        base += ["-i", url, "-vf", f"scale={self.width}:{self.height}",
                 "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        return base

    def _start(self) -> bool:
        try:
            self._proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
            return self._proc.poll() is None
        except Exception:
            self._proc = None
            return False

    def opened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def fps(self) -> float:
        return 0.0  # 管道无元数据；上层默认 25

    def read(self):
        if not self.opened() or self._proc is None or self._proc.stdout is None:
            return False, None
        buf = self._proc.stdout.read(self._frame_bytes)
        if len(buf) < self._frame_bytes:
            return False, None
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
        return True, frame

    def release(self):
        try:
            if self._proc is not None:
                if self._proc.stdout is not None:
                    self._proc.stdout.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
        except Exception:
            try:
                if self._proc is not None:
                    self._proc.kill()
            except Exception:
                pass
        self._proc = None


class VideoStream:
    """视频流读取 facade：本地文件 / RTSP / USB 索引，带断线重连与三档后端。"""

    def __init__(self, stream_url: str, *, source_backend: str = "auto",
                 resolution: tuple[int, int] | list[int] | None = None):
        self.stream_url = str(stream_url)
        self.frame_id = 0
        self.is_rtsp = self.stream_url.startswith("rtsp://") or self.stream_url.startswith("http://")
        self._reconnect_delay = 2.0
        self._max_reconnect_delay = 30.0
        self._backend_name = str(source_backend or "auto")
        self._resolution = (int(resolution[0]), int(resolution[1])) if resolution else (1280, 720)
        self._cv_failed = False
        self.backend = self._open()
        self.fps = self.backend.fps()
        if not self.fps or self.fps <= 0:
            self.fps = 25.0

    # ── 后端选择 ─────────────────────────────────────────────
    def _open(self):
        want_ffmpeg = self._backend_name == "ffmpeg" or (
            self._backend_name == "auto" and self._cv_failed)
        if not want_ffmpeg:
            backend = _CvBackend(self._resolve_url())
            if backend.opened():
                return backend
            self._cv_failed = True
            if self._backend_name == "cv":
                return backend  # 显式 cv：不降级（上层按 opened 处理）
        return _FfmpegBackend(self._resolve_url(), *self._resolution)

    def _resolve_url(self) -> str:
        # ffmpeg 管道无法读 USB 索引；cv 路径原样
        return self.stream_url

    # ── 重连（仅 RTSP；ffmpeg 重启进程，cv 重开捕获） ──────────
    def _reconnect(self) -> bool:
        if not self.is_rtsp:
            return False
        time.sleep(min(self._reconnect_delay, self._max_reconnect_delay))
        self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
        self.backend.release()
        self.backend = self._open()
        if self.backend.opened():
            self._reconnect_delay = 2.0
            return True
        return False

    def read(self):
        if not self.backend.opened():
            if self.is_rtsp and self._reconnect():
                pass
            else:
                return False, None, self.frame_id

        ok, frame = self.backend.read()
        if not ok:
            if self.is_rtsp and self._reconnect():
                ok, frame = self.backend.read()
                if ok:
                    self.frame_id += 1
                    return True, frame, self.frame_id
            return False, None, self.frame_id

        self.frame_id += 1
        self._reconnect_delay = 2.0
        return True, frame, self.frame_id

    def release(self):
        self.backend.release()

    # ── 诊断 ─────────────────────────────────────────────────
    def info(self) -> dict[str, Any]:
        return {"backend": getattr(self.backend, "name", "?"), "url": self.stream_url,
                "frames_read": self.frame_id, "fps": self.fps}
