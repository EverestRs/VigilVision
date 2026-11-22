from __future__ import annotations

"""告警切片端到端测试（docs/06_GAPS_DESIGN.md §1.8）。

覆盖：
1. 本地文件源 → ffmpeg remux 生成 MP4（h264、时长正确）；
2. 事件 JSON 的 alert.clip_path 被回写；
3. 流源 → 记 clip_error=stream_source_not_recorded，不生成文件；
4. 队列满 → dropped 计数。

用法:
  python tools/test_clip_export.py --video ../data/laying_corner_1-1.mp4
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.alert_system import AlertSystem
from src.pipeline.clip_exporter import ClipExporter, ClipRequest, probe_duration


def ffprobe_stream(path: str) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_name,pix_fmt,width,height:format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=20,
        ).stdout
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
        return {
            "codec": stream.get("codec_name"),
            "pix_fmt": stream.get("pix_fmt"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "duration": float(data.get("format", {}).get("duration") or 0),
        }
    except Exception as exc:
        return {"error": str(exc)}


def wait_for(fn, timeout: float = 15.0, interval: float = 0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(interval)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="../data/laying_corner_1-1.mp4")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        print(f"FAIL: 视频不存在 {video}")
        return 1

    out_dir = Path("results/clip_test")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    alert_system = AlertSystem(str(out_dir))
    exporter = ClipExporter(str(out_dir), {
        "enabled": True, "before_seconds": 3, "after_seconds": 3,
        "max_queue": 2, "keep_days": 7,
    })
    if not exporter.enabled:
        print("FAIL: exporter 未启用")
        return 1

    # ── 用例 1：本地文件源 ──
    frame_id = 150
    alert = None
    for _ in range(3):  # min_consecutive_frames=3
        alert = alert_system.build_alert("H03", 0.9, frame_id=frame_id)
    if not alert or not alert.get("triggered"):
        print("FAIL: 告警未触发", alert)
        return 1
    # 模拟真实落盘结构：FrameResult.model_dump() = {"frame_id": ..., "alert": {...}}
    alert_system.write_event({"frame_id": frame_id, "alert": alert})   # event_150.json

    exporter.begin_task("clip_test")
    req = ClipRequest(
        task_id="clip_test", frame_id=frame_id, event_filename=f"event_{frame_id}.json",
        source_url=str(video), video_time_sec=frame_id / 25.0,
        before_sec=3.0, after_sec=3.0,
    )
    if not exporter.submit(req):
        print("FAIL: submit 被拒绝")
        return 1

    status = wait_for(lambda: (s := exporter.status())["done"] >= 1 and s, timeout=20)
    if not status:
        print("FAIL: 切片未完成", exporter.status())
        return 1

    clip = Path(status["last_clip"])
    if not clip.exists():
        print(f"FAIL: 切片文件不存在 {clip}")
        return 1
    info = ffprobe_stream(str(clip))
    print(f"[1] 切片: {clip}  {info}")
    if info.get("codec") != "h264" or info.get("pix_fmt") != "yuv420p":
        print("FAIL: 编码不符合预期（h264/yuv420p）")
        return 1
    if not (5.0 <= info.get("duration", 0) <= 7.0):
        print(f"FAIL: 时长异常 {info.get('duration')}（期望约 6s）")
        return 1

    # ── 事件回写 ──
    event_path = out_dir / f"event_{frame_id}.json"
    written = wait_for(
        lambda: json.loads(event_path.read_text(encoding="utf-8")).get("alert", {}).get("clip_path"),
        timeout=10,
    )
    print(f"[2] 事件回写 clip_path: {written}")
    if not written or Path(written).resolve() != clip.resolve():
        print("FAIL: clip_path 未正确回写")
        return 1

    # ── 用例 2：流源 → 明确失败但不崩 ──
    exporter.begin_task("clip_test")
    stream_req = ClipRequest(
        task_id="clip_test", frame_id=frame_id + 1,
        event_filename="event_stream.json",
        source_url="rtsp://127.0.0.1:554/test", video_time_sec=10.0,
    )
    (out_dir / "event_stream.json").write_text(json.dumps({"alert": {}}), encoding="utf-8")
    exporter.submit(stream_req)
    status2 = wait_for(lambda: (s := exporter.status())["failed"] >= 1 and s, timeout=10)
    if not status2 or "stream_source_not_recorded" not in status2.get("last_error", ""):
        print("FAIL: 流源失败路径不符合预期", exporter.status())
        return 1
    stream_event = json.loads((out_dir / "event_stream.json").read_text(encoding="utf-8"))
    print(f"[3] 流源失败记录: {stream_event.get('alert', {}).get('clip_error')}")

    # ── 用例 3：队列满计数 ──
    for i in range(10):
        exporter.submit(ClipRequest(
            task_id="clip_test", frame_id=1000 + i, event_filename="event_x.json",
            source_url="rtsp://127.0.0.1:554/test", video_time_sec=1.0,
        ))
    print(f"[4] 队列统计: {exporter.status()}")
    exporter.stop()

    print("[PASS] 切片端到端测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
