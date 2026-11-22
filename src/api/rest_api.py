from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import yaml as _yaml
from pydantic import BaseModel

from src.api.websocket_server import ConnectionManager
from src.core.scene_index import SceneIndex
from src.pipeline.inference_engine import InferenceEngine
from src.pipeline.video_stream import VideoStream
from src.utils.config_loader import load_json, load_yaml
from src.utils.preflight import resolve_profile, run_preflight
from src.utils.schemas import (
    AnalyzeStreamRequest,
    AnalyzeStreamResponse,
    EvaluateRequest,
    TrainRequest,
    VLMReviewRequest,
    VLMReviewResponse,
)
from src.pipeline.video_writer import open_video_writer

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"

raw_model_cfg = load_yaml(CONFIG_DIR / "model_config.yaml")
vlm_cfg = load_yaml(CONFIG_DIR / "vlm_config.yaml")
behavior_rules = load_json(CONFIG_DIR / "behavior_rules.json")

if isinstance(behavior_rules, dict) and isinstance(behavior_rules.get("heuristics"), dict):
    raw_model_cfg["behavior_heuristics"] = dict(behavior_rules["heuristics"])
if isinstance(behavior_rules, dict) and isinstance(behavior_rules.get("alert_policy"), dict):
    raw_model_cfg["alert_policy"] = dict(behavior_rules["alert_policy"])


def apply_inference_profile(config: dict, profile_name: str | None = None) -> tuple[dict, str, dict]:
    selected_profile = profile_name or os.getenv("INFERENCE_PROFILE", "realtime")
    merged, resolved_model_paths = resolve_profile(config, selected_profile)
    return merged, selected_profile, resolved_model_paths


model_cfg, active_profile, active_model_paths = apply_inference_profile(raw_model_cfg)


def _resolve_replay_buffer_frames(config: dict) -> int:
    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    return int(max(0, inference_cfg.get("replay_buffer_frames", 24)))


def _estimate_task_replay_history_size(config: dict, duration_seconds: int) -> int:
    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    replay_from_start = bool(inference_cfg.get("replay_from_task_start", True))
    base_buffer = int(max(0, inference_cfg.get("replay_buffer_frames", 24)))
    if not replay_from_start:
        return base_buffer

    max_fps = float(max(0.0, inference_cfg.get("max_fps", 0)))
    push_stride = int(max(1, inference_cfg.get("push_stride", 1)))
    if duration_seconds <= 0 or max_fps <= 0:
        return max(base_buffer, 300)

    estimated = int(duration_seconds * max_fps / push_stride) + 16
    return int(max(base_buffer, min(5000, estimated)))

app = FastAPI(title="Campus Security Agent API", version="0.1.0")
manager = ConnectionManager(history_size=_resolve_replay_buffer_frames(model_cfg))
engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(BASE_DIR / "results"))
TASKS: dict[str, dict] = {}
EVAL_JOBS: dict[str, dict] = {}
TRAIN_JOBS: dict[str, dict] = {"_active": {"running": False, "train_id": ""}}
scene_index = SceneIndex(vlm_cfg, results_dir=str(BASE_DIR / "results"))


def reconfigure_profile(profile_name: str) -> dict:
    global model_cfg
    global active_profile
    global active_model_paths
    global engine

    available_profiles = raw_model_cfg.get("profiles", {})
    if profile_name not in available_profiles:
        raise HTTPException(status_code=400, detail=f"invalid profile: {profile_name}")

    model_cfg, active_profile, active_model_paths = apply_inference_profile(raw_model_cfg, profile_name=profile_name)
    manager.set_history_size(_resolve_replay_buffer_frames(model_cfg))
    old_engine = engine
    engine = InferenceEngine(model_cfg=model_cfg, vlm_cfg=vlm_cfg, output_dir=str(BASE_DIR / "results"))
    old_engine.close()
    return {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "inference": model_cfg.get("inference", {}),
        "models": active_model_paths,
    }


def encode_frame_to_base64(frame, jpeg_quality: int = 75) -> str:
    quality = int(max(30, min(95, jpeg_quality)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def annotate_frame(frame, payload: dict, show_pose_overlay: bool = True) -> any:
    detections = payload.get("detections", [])
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x, y, w, h = [int(value) for value in bbox]
        behavior = det.get("behavior", {})
        behavior_type = behavior.get("type", "Normal")
        confidence = float(behavior.get("confidence", 0.0))
        role = behavior.get("role", "unknown")
        risk = behavior.get("risk_level", "LOW")
        track_id = det.get("track_id", "P000")

        if risk == "HIGH":
            color = (0, 0, 255)
        elif risk == "MEDIUM":
            color = (0, 165, 255)
        else:
            color = (0, 200, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{track_id} {behavior_type} {confidence:.2f} {role}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_top = max(0, y - text_h - 8)
        cv2.rectangle(frame, (x, text_bg_top), (x + text_w + 6, y), color, -1)
        cv2.putText(
            frame,
            label,
            (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if show_pose_overlay:
            for keypoint in det.get("pose_keypoints", []):
                if len(keypoint) < 2:
                    continue
                kp_x = int(keypoint[0])
                kp_y = int(keypoint[1])
                conf = float(keypoint[2]) if len(keypoint) > 2 else 1.0
                if conf < 0.2:
                    continue
                cv2.circle(frame, (kp_x, kp_y), 2, (255, 255, 0), -1)

    alert = payload.get("alert", {})
    if alert.get("triggered"):
        alert_text = f"ALERT: {alert.get('type', 'unknown')}"
        cv2.rectangle(frame, (10, 10), (420, 48), (0, 0, 255), -1)
        cv2.putText(frame, alert_text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def visible_signature(payload: dict, bbox_epsilon_px: float = 2.0,
                      conf_epsilon: float = 0.05) -> tuple:
    """WS 差分推送的“可见签名”。

    只把前端能看见的变化纳入签名：bbox（按像素粒度量化）、行为标签/置信度/风险级、
    告警状态、命中的规则集合。命中相同签名的帧无需重发（前端靠插值补间）。
    """
    bbox_step = max(1.0, float(bbox_epsilon_px))
    conf_step = max(0.001, float(conf_epsilon))
    detections = []
    for det in payload.get("detections") or []:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        behavior = det.get("behavior") or {}
        try:
            quantized = [int(round(float(value) / bbox_step)) for value in bbox[:4]]
        except Exception:
            quantized = [0, 0, 0, 0]
        detections.append((
            str(det.get("track_id", "")),
            quantized[0], quantized[1], quantized[2], quantized[3],
            str(behavior.get("type", "")),
            str(behavior.get("risk_level", "")),
            str(behavior.get("role", "")),
            int(round(float(behavior.get("confidence", 0.0)) / conf_step)),
        ))
    detections.sort()
    alert = payload.get("alert") or {}
    rule_ids = tuple(sorted(str(rule.get("rule_id", "")) for rule in (payload.get("rule_alerts") or [])))
    return (tuple(detections), bool(alert.get("triggered")), str(alert.get("type", "")), rule_ids)


def should_skip_diff_push(signature: tuple, last_signature: tuple | None,
                          now_ts: float, last_push_ts: float,
                          heartbeat_seconds: float, alert_now: bool) -> bool:
    """是否跳过本次 WS 推送。

    规则优先级：告警帧必发 > 心跳到期必发 > 可见签名相同才跳过。
    心跳用墙钟而非 frame_id：pipeline 的 drain 结果帧会乱序/滞后，
    用帧号差判断会导致心跳永不触发、实时结果被全部跳过。
    """
    if alert_now:
        return False
    if (float(now_ts) - float(last_push_ts)) >= max(0.05, float(heartbeat_seconds)):
        return False
    return signature == last_signature


def read_alert_logs(results_dir: Path, limit: int = 50) -> list[dict]:
    files = sorted(results_dir.glob("event_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    alerts: list[dict] = []
    for file in files[:limit]:
        try:
            alerts.append(json.loads(file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return alerts


def run_render_task(task_id: str, stream_url: str) -> None:
    inference_cfg = model_cfg.get("inference", {})
    show_pose_overlay = bool(inference_cfg.get("show_pose_overlay", True))
    output_dir = BASE_DIR / "results" / "rendered"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_id}.mp4"

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = "cannot_open_stream"
        return

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1.0:
        source_fps = 25.0
    engine.begin_task(task_id)
    engine.set_source_url(stream_url)
    engine.set_source_fps(source_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = "invalid_video_size"
        cap.release()
        return

    writer = open_video_writer(str(output_path), source_fps, (width, height))
    if not writer.isOpened():
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = f"cannot_open_video_writer: {getattr(writer, 'last_error', '')}"
        cap.release()
        return

    TASKS[task_id]["status"] = "running"
    TASKS[task_id]["mode"] = "render"
    TASKS[task_id]["render"] = {
        "frames_total": total_frames,
        "frames_processed": 0,
        "progress": 0.0,
        "fps": round(source_fps, 3),
        "codec": getattr(writer, "codec", ""),
        "output_video_path": str(output_path),
    }

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            result = engine.process(frame, frame_id)
            payload = result.model_dump()
            vis_frame = annotate_frame(frame.copy(), payload, show_pose_overlay=show_pose_overlay)
            writer.write(vis_frame)

            progress = 0.0
            if total_frames > 0:
                progress = min(1.0, frame_id / total_frames)

            TASKS[task_id]["last_frame"] = frame_id
            TASKS[task_id]["last_alert"] = payload.get("alert", {})
            TASKS[task_id]["render"] = {
                "frames_total": total_frames,
                "frames_processed": frame_id,
                "progress": round(progress, 4),
                "fps": round(source_fps, 3),
                "codec": getattr(writer, "codec", ""),
                "output_video_path": str(output_path),
            }

        TASKS[task_id]["status"] = "finished"
    except Exception as exc:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = f"render_failed: {exc}"
    finally:
        cap.release()
        writer.release()
        # ffmpeg 退出码非 0（磁盘满/参数错）时不能报完成——避免"文件坏但任务成功"
        writer_error = getattr(writer, "last_error", "")
        if writer_error and TASKS[task_id].get("status") == "finished":
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["error"] = f"video_writer: {writer_error}"


async def run_task(task_id: str, stream_url: str, duration: int):
    inference_cfg = model_cfg.get("inference", {})
    # T2.2: 解码三档（cv/ffmpeg/auto）——g2 探针实测选档后写入档位；失败回退默认 cv
    try:
        stream = VideoStream(
            stream_url,
            source_backend=str(inference_cfg.get("source_backend", "auto")),
            resolution=inference_cfg.get("source_resolution") or None,
        )
    except Exception:
        stream = VideoStream(stream_url)
    frame_stride = int(max(1, inference_cfg.get("frame_stride", 1)))
    push_stride = int(max(1, inference_cfg.get("push_stride", 1)))
    send_frame_image_b64 = bool(inference_cfg.get("send_frame_image_b64", False))
    max_fps = float(max(0.0, inference_cfg.get("max_fps", 0)))
    startup_no_drop_seconds = float(max(0.0, min(5.0, inference_cfg.get("startup_no_drop_seconds", 1.5))))
    jpeg_quality = int(inference_cfg.get("jpeg_quality", 75))
    ws_sleep_ms = int(max(0, inference_cfg.get("ws_sleep_ms", 1)))
    # 差分推送：可见内容未变化的帧不再重发（前端用现有插值补间），告警/心跳帧强制发
    diff_cfg = dict(inference_cfg.get("ws_diff_push", {}) or {})
    diff_enabled = bool(diff_cfg.get("enabled", False))
    diff_bbox_eps = float(diff_cfg.get("bbox_epsilon_px", 2.0))
    # 置信度量化粒度取 0.05：0.01 级的抖动是复核噪声，0.05 以上的变化才值得重绘
    diff_conf_eps = float(diff_cfg.get("conf_epsilon", 0.05))
    diff_heartbeat_sec = max(0.05, float(diff_cfg.get("heartbeat_seconds", 0.5)))
    diff_skipped = 0
    last_push_signature = None
    last_push_ts = 0.0
    source_resize = bool(inference_cfg.get("source_resize", False))
    send_pose_keypoints = bool(inference_cfg.get("send_pose_keypoints", True))
    show_pose_overlay = bool(inference_cfg.get("show_pose_overlay", True))
    target_size = inference_cfg.get("input_size", [960, 540])
    target_width = int(target_size[0]) if isinstance(target_size, list) and len(target_size) > 0 else 960
    target_height = int(target_size[1]) if isinstance(target_size, list) and len(target_size) > 1 else 540

    source_fps = float(stream.fps if stream.fps and stream.fps > 0 else 25.0)
    TASKS[task_id]["source_fps"] = round(source_fps, 3)
    engine.begin_task(task_id)
    engine.set_source_url(stream_url)
    engine.set_source_fps(source_fps)
    max_frames = int(duration * source_fps) if duration > 0 else None
    min_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
    is_local_file = Path(stream_url).exists()
    read_min_interval = (1.0 / source_fps) if is_local_file and source_fps > 0 else 0.0
    stop_event = threading.Event()
    task_start_ts = time.perf_counter()

    decode_queue: queue.Queue = queue.Queue(maxsize=2)
    result_queue: queue.Queue = queue.Queue(maxsize=2)
    packet_queue: queue.Queue = queue.Queue(maxsize=2)

    stats_lock = threading.Lock()
    read_counter = 0
    infer_counter = 0
    engine_process_counter = 0
    replay_counter = 0
    encoded_counter = 0
    pushed_count = 0
    dropped_frames = 0
    dropped_results = 0
    dropped_packets = 0
    last_processed_frame_id = 0
    action_runs_total = 0
    action_skip_warmup_total = 0
    action_skip_track_age_total = 0
    action_skip_gate_total = 0
    action_skip_cached_total = 0
    action_skip_default_total = 0

    stage_a_ms_total = 0.0
    stage_a_ms_max = 0.0
    stage_a_count = 0

    stage_b_ms_total = 0.0
    stage_b_ms_max = 0.0
    stage_b_count = 0

    stage_c_ms_total = 0.0
    stage_c_ms_max = 0.0
    stage_c_count = 0

    stage_d_ms_total = 0.0
    stage_d_ms_max = 0.0
    stage_d_count = 0
    stage_d_sleep_ms_total = 0.0

    thread_error = ""

    def enqueue_latest(q: queue.Queue, item: object, drop_counter_name: str) -> None:
        nonlocal dropped_frames
        nonlocal dropped_results
        nonlocal dropped_packets

        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            q.get_nowait()
        except Exception:
            pass

        if drop_counter_name == "frames":
            dropped_frames += 1
        elif drop_counter_name == "results":
            dropped_results += 1
        elif drop_counter_name == "packets":
            dropped_packets += 1

        try:
            q.put_nowait(item)
        except Exception:
            pass

    # B6: 管道模式开关（send_frame_image_b64=True 时帧与结果必须同步，回退串行）
    pipeline_mode = bool(getattr(engine, "_pipeline_enabled", False)) and not send_frame_image_b64

    # C5: 事件级评估——逐帧标签落盘（eval_dump 任务才开启）
    _eval_fh = None
    if TASKS.get(task_id, {}).get("eval_dump"):
        _dump_dir = BASE_DIR / "results" / "eval_dumps"
        _dump_dir.mkdir(parents=True, exist_ok=True)
        _eval_fh = open(_dump_dir / f"{task_id}.jsonl", "w", encoding="utf-8")

    def build_payload(result) -> dict:
        """FrameResult → WS payload（B6: 同步/管道两路共用）。"""
        payload = result.model_dump()
        if _eval_fh is not None:
            try:
                _eval_fh.write(json.dumps({
                    "frame_id": payload.get("frame_id"),
                    "labels": [[d.get("track_id"), (d.get("behavior") or {}).get("type")]
                               for d in payload.get("detections", [])],
                    "alert_triggered": bool(payload.get("alert", {}).get("triggered")),
                    "alert_type": payload.get("alert", {}).get("type", "none"),
                    "rule_alerts": [ra.get("matched_class") for ra in payload.get("rule_alerts", [])
                                    if isinstance(ra, dict) and ra.get("matched_class")],
                    "sources": payload.get("metrics", {}).get("label_sources", {}),
                    # 诊断：告警判定输入（main_label/main_confidence）与逐目标置信度，
                    # 用于区分"规则没命中"和"命中但被连续帧/置信度门槛挡掉"
                    "alert": payload.get("alert", {}),
                    "main": [payload.get("metrics", {}).get("main_label"),
                             payload.get("metrics", {}).get("main_confidence")],
                    "confs": [[d.get("track_id"),
                               round(float((d.get("behavior") or {}).get("confidence", 0.0)), 3)]
                              for d in payload.get("detections", [])],
                }, ensure_ascii=False) + "\n")
                _eval_fh.flush()
            except Exception:
                pass
        action_debug = payload.get("metrics", {}).get("action_debug", {})
        if isinstance(action_debug, dict):
            TASKS[task_id]["action_perf"] = {
                "action_runs": int(action_debug.get("action_runs", 0)),
                "action_skip_warmup": int(action_debug.get("action_skip_warmup", 0)),
                "action_skip_track_age": int(action_debug.get("action_skip_track_age", 0)),
                "action_skip_gate": int(action_debug.get("action_skip_gate", 0)),
                "action_skip_cached": int(action_debug.get("action_skip_cached", 0)),
                "action_skip_default": int(action_debug.get("action_skip_default", 0)),
                "action_stride": int(action_debug.get("action_stride", 0)),
                "max_action_tracks": int(action_debug.get("max_action_tracks", 0)),
                "action_warmup_frames": int(action_debug.get("action_warmup_frames", 0)),
                "action_min_track_age": int(action_debug.get("action_min_track_age", 0)),
            }
        if not send_pose_keypoints:
            for item in payload.get("detections", []):
                item["pose_keypoints"] = []
        payload.setdefault("metrics", {})["streaming"] = {
            "frame_stride": frame_stride,
            "push_stride": push_stride,
            "source_fps": round(source_fps, 3),
            "jpeg_quality": jpeg_quality,
            "ws_sleep_ms": ws_sleep_ms,
            "source_resize": source_resize,
            "send_pose_keypoints": send_pose_keypoints,
            "show_pose_overlay": show_pose_overlay,
            "input_size": [target_width, target_height],
        }
        return payload

    def reader_loop() -> None:
        nonlocal thread_error
        nonlocal read_counter
        nonlocal stage_a_ms_total
        nonlocal stage_a_ms_max
        nonlocal stage_a_count
        last_read_ts = 0.0
        try:
            while (
                not stop_event.is_set()
                and not TASKS.get(task_id, {}).get("stop_requested", False)
                and (max_frames is None or read_counter < max_frames)
            ):
                if read_min_interval > 0:
                    now = time.perf_counter()
                    elapsed = now - last_read_ts
                    if elapsed < read_min_interval:
                        time.sleep(read_min_interval - elapsed)
                    last_read_ts = time.perf_counter()

                stage_a_t0 = time.perf_counter()
                ok, frame, frame_id = stream.read()
                if not ok:
                    break

                read_counter += 1
                orig_frame = frame  # 问题1：原帧复核通道——降采样前的原始帧
                if source_resize and target_width > 0 and target_height > 0:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

                while not stop_event.is_set():
                    in_startup_protection = (time.perf_counter() - task_start_ts) < startup_no_drop_seconds
                    if in_startup_protection and decode_queue.full():
                        time.sleep(0.001)
                        continue

                    with stats_lock:
                        enqueue_latest(decode_queue, (orig_frame, frame, frame_id), "frames")
                        stage_a_ms = (time.perf_counter() - stage_a_t0) * 1000.0
                        stage_a_ms_total += stage_a_ms
                        stage_a_ms_max = max(stage_a_ms_max, stage_a_ms)
                        stage_a_count += 1
                    break
        except Exception as exc:
            thread_error = f"reader_failed: {exc}"
        finally:
            stop_event.set()

    def push_payload(payload: dict, push_frame_id: int, frame=None) -> None:
        """差分 + 入队：pipeline / 同步 / replay 三个分支共用，避免绕过差分判断。"""
        nonlocal diff_skipped, last_push_signature, last_push_ts
        if diff_enabled:
            now_ts = time.perf_counter()
            signature = visible_signature(payload, diff_bbox_eps, diff_conf_eps)
            alert_now = (bool((payload.get("alert") or {}).get("triggered"))
                         or bool(payload.get("rule_alerts")))
            if should_skip_diff_push(signature, last_push_signature, now_ts,
                                     last_push_ts, diff_heartbeat_sec, alert_now):
                diff_skipped += 1
                return
            last_push_signature = signature
            last_push_ts = now_ts
        with stats_lock:
            enqueue_latest(result_queue, (frame, push_frame_id, payload), "results")

    def infer_loop() -> None:
        nonlocal thread_error
        nonlocal infer_counter
        nonlocal engine_process_counter
        nonlocal replay_counter
        nonlocal last_processed_frame_id
        nonlocal stage_b_ms_total
        nonlocal stage_b_ms_max
        nonlocal stage_b_count
        last_payload: dict | None = None
        last_infer_ts = 0.0
        try:
            while not stop_event.is_set() or not decode_queue.empty():
                if TASKS.get(task_id, {}).get("stop_requested", False):
                    stop_event.set()
                    break
                try:
                    orig_frame, frame, frame_id = decode_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                stage_b_t0 = time.perf_counter()
                should_infer = frame_id == 1 or (frame_id % frame_stride) == 0
                should_push = frame_id == 1 or (frame_id % push_stride) == 0

                # max_fps 语义 = 推理帧率上限，只对真正送推理的帧限速。
                # 原实现放在每个解码帧上，与 frame_stride 相乘：8fps / 4 = 2fps，
                # 实测任务只有 1.8fps，而引擎单帧仅 ~80ms（12.6fps）——瓶颈是节流，不是模型。
                if should_infer and min_interval > 0:
                    now = time.perf_counter()
                    elapsed = now - last_infer_ts
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                    last_infer_ts = time.perf_counter()

                if should_infer and pipeline_mode:
                    # B6: 投递本帧 → 收割已完成结果（结果滞后数帧，payload 带各自 frame_id）
                    engine.submit_frame(frame, frame_id, orig_frame=orig_frame)
                    drained = engine.drain_results()
                    for r in drained:
                        p = build_payload(r)
                        last_payload = p
                        engine_process_counter += 1
                        if (r.frame_id % push_stride) == 0 or r.frame_id == 1:
                            push_payload(p, r.frame_id, None)
                    if drained:
                        infer_counter += 1
                        last_processed_frame_id = frame_id
                        continue
                    # 结果未就绪：复用最近 payload（同重放语义），无历史则跳过本帧
                    if last_payload is None:
                        continue
                    payload = dict(last_payload)
                    payload["frame_id"] = frame_id
                    replay_counter += 1
                elif should_infer:
                    result = engine.process(frame, frame_id, orig_frame=orig_frame)
                    payload = build_payload(result)
                    engine_process_counter += 1
                    last_payload = payload
                elif last_payload is not None:
                    # 结果未就绪：不推送复用的旧结果。
                    # 旧 bbox 配上新 frame_id 会让前端按视频时间选中的“最新事件”
                    # 始终是旧位置 —— 视觉上框完全不动（实测 8s 任务 112 帧如此）。
                    # 前端靠真实结果之间的插值补间，这里只维持处理节奏。
                    replay_counter += 1
                    with stats_lock:
                        stage_b_ms = (time.perf_counter() - stage_b_t0) * 1000.0
                        stage_b_ms_total += stage_b_ms
                        stage_b_ms_max = max(stage_b_ms_max, stage_b_ms)
                        stage_b_count += 1
                    continue
                else:
                    continue

                infer_counter += 1
                last_processed_frame_id = frame_id

                if not should_push:
                    with stats_lock:
                        stage_b_ms = (time.perf_counter() - stage_b_t0) * 1000.0
                        stage_b_ms_total += stage_b_ms
                        stage_b_ms_max = max(stage_b_ms_max, stage_b_ms)
                        stage_b_count += 1
                    continue

                push_payload(payload, frame_id, frame)
                with stats_lock:
                    stage_b_ms = (time.perf_counter() - stage_b_t0) * 1000.0
                    stage_b_ms_total += stage_b_ms
                    stage_b_ms_max = max(stage_b_ms_max, stage_b_ms)
                    stage_b_count += 1
        except Exception as exc:
            thread_error = f"inference_failed: {exc}"
        finally:
            stop_event.set()

    def encode_loop() -> None:
        nonlocal thread_error
        nonlocal encoded_counter
        nonlocal stage_c_ms_total
        nonlocal stage_c_ms_max
        nonlocal stage_c_count
        try:
            while not stop_event.is_set() or not result_queue.empty():
                if TASKS.get(task_id, {}).get("stop_requested", False):
                    stop_event.set()
                    break
                try:
                    frame, frame_id, payload = result_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                stage_c_t0 = time.perf_counter()
                if send_frame_image_b64:
                    vis_frame = annotate_frame(frame.copy(), payload, show_pose_overlay=show_pose_overlay)
                    payload["frame_image_b64"] = encode_frame_to_base64(vis_frame, jpeg_quality=jpeg_quality)
                else:
                    payload.pop("frame_image_b64", None)
                encoded_counter += 1

                with stats_lock:
                    enqueue_latest(packet_queue, (frame_id, payload), "packets")
                    stage_c_ms = (time.perf_counter() - stage_c_t0) * 1000.0
                    stage_c_ms_total += stage_c_ms
                    stage_c_ms_max = max(stage_c_ms_max, stage_c_ms)
                    stage_c_count += 1
        except Exception as exc:
            thread_error = f"encode_failed: {exc}"
        finally:
            stop_event.set()

    reader_thread = threading.Thread(target=reader_loop, name=f"reader-{task_id}", daemon=True)
    infer_thread = threading.Thread(target=infer_loop, name=f"infer-{task_id}", daemon=True)
    encode_thread = threading.Thread(target=encode_loop, name=f"encode-{task_id}", daemon=True)

    TASKS[task_id]["status"] = "running"
    try:
        reader_thread.start()
        infer_thread.start()
        encode_thread.start()

        while True:
            if TASKS.get(task_id, {}).get("stop_requested", False):
                TASKS[task_id]["status"] = "stopped"
                stop_event.set()
                break

            if thread_error:
                raise RuntimeError(thread_error)

            has_live_worker = reader_thread.is_alive() or infer_thread.is_alive() or encode_thread.is_alive()
            if not has_live_worker and packet_queue.empty():
                break

            try:
                frame_id, payload = packet_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue

            pushed_count += 1
            stage_d_t0 = time.perf_counter()
            TASKS[task_id]["last_frame"] = frame_id
            TASKS[task_id]["last_alert"] = payload.get("alert", {})
            action_debug = payload.get("metrics", {}).get("action_debug", {})
            action_runs_total += int(action_debug.get("action_runs", 0))
            action_skip_warmup_total += int(action_debug.get("action_skip_warmup", 0))
            action_skip_track_age_total += int(action_debug.get("action_skip_track_age", 0))
            action_skip_gate_total += int(action_debug.get("action_skip_gate", 0))
            action_skip_cached_total += int(action_debug.get("action_skip_cached", 0))
            action_skip_default_total += int(action_debug.get("action_skip_default", 0))
            # P0-3 可观测：NL 规则来源约束状态与被拦截来源计数（排障/验收用）
            nl_status = payload.get("metrics", {}).get("scene_debug", {}).get("nl_rule_engine")
            if isinstance(nl_status, dict):
                TASKS[task_id]["nl_rule_status"] = nl_status
            with stats_lock:
                stage_a_avg = stage_a_ms_total / stage_a_count if stage_a_count > 0 else 0.0
                stage_b_avg = stage_b_ms_total / stage_b_count if stage_b_count > 0 else 0.0
                stage_c_avg = stage_c_ms_total / stage_c_count if stage_c_count > 0 else 0.0
                stage_d_avg = stage_d_ms_total / stage_d_count if stage_d_count > 0 else 0.0
            TASKS[task_id]["stream_stats"] = {
                "source_fps": round(source_fps, 3),
                "read_frames": read_counter,
                "inference_frames": infer_counter,
                "engine_process_frames": engine_process_counter,
                "replayed_frames": replay_counter,
                "encoded_frames": encoded_counter,
                "pushed_frames": pushed_count,
                "ws_diff_skipped": diff_skipped,
                "last_processed_frame_id": last_processed_frame_id,
                "dropped_frames_decode": dropped_frames,
                "dropped_frames_result": dropped_results,
                "dropped_frames_packet": dropped_packets,
                "max_fps": max_fps,
                "startup_no_drop_seconds": startup_no_drop_seconds,
                "frame_stride": frame_stride,
                "push_stride": push_stride,
                "replay_buffer_frames": manager.get_task_history_size(task_id),
                "action_perf": {
                    "action_runs_total": action_runs_total,
                    "action_skip_warmup_total": action_skip_warmup_total,
                    "action_skip_track_age_total": action_skip_track_age_total,
                    "action_skip_gate_total": action_skip_gate_total,
                    "action_skip_cached_total": action_skip_cached_total,
                    "action_skip_default_total": action_skip_default_total,
                },
                "stage_ms": {
                    "stage_a_decode": {
                        "avg_ms": round(stage_a_avg, 3),
                        "max_ms": round(stage_a_ms_max, 3),
                        "count": stage_a_count,
                    },
                    "stage_b_inference": {
                        "avg_ms": round(stage_b_avg, 3),
                        "max_ms": round(stage_b_ms_max, 3),
                        "count": stage_b_count,
                    },
                    "stage_c_encode": {
                        "avg_ms": round(stage_c_avg, 3),
                        "max_ms": round(stage_c_ms_max, 3),
                        "count": stage_c_count,
                    },
                    "stage_d_push": {
                        "avg_ms": round(stage_d_avg, 3),
                        "max_ms": round(stage_d_ms_max, 3),
                        "count": stage_d_count,
                        "sleep_ms_total": round(stage_d_sleep_ms_total, 3),
                    },
                },
            }

            payload_metrics = payload.setdefault("metrics", {})
            payload_metrics["stream_runtime"] = {
                "read_frames": read_counter,
                "inference_frames": infer_counter,
                "engine_process_frames": engine_process_counter,
                "replayed_frames": replay_counter,
                "encoded_frames": encoded_counter,
                "pushed_frames": pushed_count,
                "ws_diff_skipped": diff_skipped,
                "last_processed_frame_id": last_processed_frame_id,
                "task_status": TASKS[task_id].get("status", "running"),
            }

            manager.record_message(task_id, payload)
            # 快照已由 AlertSystem.build_alert 在引擎侧落盘（frame 在触发点可用）；
            # 此处旧实现引用读帧线程局部变量 frame，NameError 被吞——快照链路断裂（A-1 修复）

            await manager.broadcast(task_id, payload)
            stage_d_ms = (time.perf_counter() - stage_d_t0) * 1000.0
            with stats_lock:
                stage_d_ms_total += stage_d_ms
                stage_d_ms_max = max(stage_d_ms_max, stage_d_ms)
                stage_d_count += 1
            if ws_sleep_ms > 0:
                await asyncio.sleep(ws_sleep_ms / 1000.0)
                with stats_lock:
                    stage_d_sleep_ms_total += float(ws_sleep_ms)

        if TASKS.get(task_id, {}).get("status") != "stopped":
            TASKS[task_id]["status"] = "finished"
    except Exception as exc:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(exc)
    finally:
        stop_event.set()
        if _eval_fh is not None:
            try:
                _eval_fh.close()
            except Exception:
                pass
        if reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        if infer_thread.is_alive():
            infer_thread.join(timeout=1.0)
        if encode_thread.is_alive():
            encode_thread.join(timeout=1.0)
        stream.release()
        manager.connections.pop(task_id, None)


@app.get("/health")
def health_check():
    runtime = model_cfg.get("runtime", {})
    inference = model_cfg.get("inference", {})
    preflight = run_preflight(BASE_DIR, raw_model_cfg, active_profile)
    return {
        "status": "ok",
        "profile": active_profile,
        "available_profiles": list(raw_model_cfg.get("profiles", {}).keys()),
        "runtime": runtime,
        "inference": inference,
        "models": active_model_paths,
        "preflight_summary": preflight.get("summary", {}),
        "detector_debug": engine.detector.get_debug_info(),
        "pose_debug": engine.pose_estimator.get_debug_info(),
        "action_debug": engine.action_recognizer.get_debug_info(),
    }


@app.get("/api/v1/preflight")
def preflight(profile: str | None = None):
    target_profile = profile or active_profile
    return run_preflight(BASE_DIR, raw_model_cfg, target_profile)


@app.get("/api/v1/debug/detector")
def detector_debug():
    return {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "detector": engine.detector.get_debug_info(),
    }


@app.get("/api/v1/debug/pose")
def pose_debug():
    return {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "pose": engine.pose_estimator.get_debug_info(),
    }


@app.get("/api/v1/debug/action")
def action_debug():
    return {
        "profile": active_profile,
        "runtime": model_cfg.get("runtime", {}),
        "action": engine.action_recognizer.get_debug_info(),
        "vlm_cache": engine._vlm_cache.stats(),
        "vlm_router": engine.vlm_router.stats(),
        "pipeline": engine.pipeline_stats(),
        "smooth_window": engine._smooth_window,
    }


@app.post("/api/v1/profile")
def set_profile(profile: str):
    if any(task.get("status") == "running" for task in TASKS.values()):
        raise HTTPException(status_code=409, detail="Cannot switch profile while a task is running")
    return {"status": "ok", **reconfigure_profile(profile)}


@app.post("/api/v1/analyze/stream", response_model=AnalyzeStreamResponse)
async def analyze_stream(request: AnalyzeStreamRequest):
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    replay_history_size = _estimate_task_replay_history_size(model_cfg, int(request.duration))
    manager.prepare_task(task_id, history_size=replay_history_size)
    TASKS[task_id] = {
        "status": "queued",
        "stream_url": request.stream_url,
        "last_frame": 0,
        "stop_requested": False,
        "eval_dump": bool(getattr(request, "eval_dump", False)),
    }
    asyncio.create_task(run_task(task_id, request.stream_url, request.duration))
    return AnalyzeStreamResponse(
        task_id=task_id,
        status="running",
        endpoint=f"ws://127.0.0.1:8000/ws/task/{task_id}",
    )


@app.post("/api/v1/analyze/render")
async def analyze_render(request: AnalyzeStreamRequest):
    task_id = f"render_{uuid.uuid4().hex[:8]}"
    TASKS[task_id] = {
        "status": "queued",
        "mode": "render",
        "stream_url": request.stream_url,
        "last_frame": 0,
        "render": {
            "frames_total": 0,
            "frames_processed": 0,
            "progress": 0.0,
            "fps": 0.0,
            "output_video_path": "",
        },
    }
    thread = threading.Thread(target=run_render_task, args=(task_id, request.stream_url), daemon=True)
    thread.start()
    return {
        "task_id": task_id,
        "status": "running",
        "endpoint": f"/api/v1/tasks/{task_id}",
    }


@app.websocket("/ws/task/{task_id}")
async def ws_task(websocket: WebSocket, task_id: str):
    await manager.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id, websocket)


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str):
    return TASKS.get(task_id, {"status": "not_found"})


@app.post("/api/v1/tasks/{task_id}/stop")
def stop_task(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")

    status = task.get("status", "unknown")
    if status in {"finished", "failed", "stopped"}:
        task["status"] = "stopped"
        task["stop_requested"] = True
        return {"task_id": task_id, "status": "stopped", "detail": "task already ended"}

    task["stop_requested"] = True
    task["status"] = "stopped"
    return {"task_id": task_id, "status": "stopped", "detail": "stop requested"}


@app.get("/api/v1/events")
def get_events(limit: int = 50):
    """查询历史预警事件（与 /api/v1/alerts 同源）"""
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"items": read_alert_logs(results_dir, limit=limit)}


@app.get("/api/v1/scene/log")
def get_scene_log(limit: int = 50, task_id: str | None = None,
                  important_only: bool = False, since_frame_id: int = 0):
    """查询场景叙述日志（默认当前任务；可指定 task_id 查询历史任务）。

    important_only: 只返回 importance=high 的条目（前端"重要理解"面板/通知用）
    since_frame_id: 只返回帧号大于该值的条目（前端自动刷新做增量，避免重复提示）
    """
    items = engine.scene_narrator.get_log(limit=limit, task_id=task_id)
    if important_only:
        items = [item for item in items if str(item.get("importance", "")) == "high"]
    if since_frame_id > 0:
        items = [item for item in items if int(item.get("frame_id", 0) or 0) > int(since_frame_id)]
    return {"items": items,
            "current_task_id": engine.scene_narrator.current_task_id,
            "latest_frame_id": engine.scene_narrator.status().get("latest_frame_id", 0)}


@app.get("/api/v1/scene/status")
def get_scene_status():
    """场景叙述器 / 告警裁决器运行状态（用于前端健康面板）"""
    return {
        "narrator": engine.scene_narrator.status(),
        "adjudicator": engine.alert_adjudicator.status(),
        "frame_buffer_frames": len(engine.frame_buffer),
    }


@app.get("/api/v1/search")
def search_scene_index(
    q: str = Query(..., min_length=1, description="自然语言检索词"),
    top_k: int = Query(10, ge=1, le=50),
    doc_type: str | None = Query(None, description="scene_narration | alert_event | behavior_prototype"),
    task_id: str | None = Query(None, description="按任务过滤"),
):
    """对场景叙述与告警事件做自然语言检索（BM25 + 可选向量重排）"""
    return scene_index.search(q, top_k=top_k, doc_type=doc_type, task_id=task_id)


@app.get("/api/v1/search/status")
def search_scene_status():
    return scene_index.status()


class AlertRuleCreateRequest(BaseModel):
    text: str


class AlertRuleUpdateRequest(BaseModel):
    text: str | None = None
    enabled: bool | None = None
    min_confidence: float | None = None


@app.post("/api/v1/alert-rules")
def create_alert_rule(request: AlertRuleCreateRequest):
    """创建自然语言告警规则（同步调用 VLM 解析，约 2-4 秒）"""
    try:
        rule = engine.nl_rule_engine.add_rule(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"规则解析失败（VLM 服务不可用？）: {exc}")
    return rule


@app.get("/api/v1/alert-rules")
def list_alert_rules():
    return {"items": engine.nl_rule_engine.list_rules(), "status": engine.nl_rule_engine.status()}


@app.get("/api/v1/alert-rules/{rule_id}")
def get_alert_rule(rule_id: str):
    rule = engine.nl_rule_engine.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"rule not found: {rule_id}")
    return rule


@app.put("/api/v1/alert-rules/{rule_id}")
def update_alert_rule(rule_id: str, request: AlertRuleUpdateRequest):
    try:
        return engine.nl_rule_engine.update_rule(
            rule_id, text=request.text, enabled=request.enabled, min_confidence=request.min_confidence
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"rule not found: {rule_id}")


@app.delete("/api/v1/alert-rules/{rule_id}")
def delete_alert_rule(rule_id: str):
    if not engine.nl_rule_engine.remove_rule(rule_id):
        raise HTTPException(status_code=404, detail=f"rule not found: {rule_id}")
    return {"status": "deleted", "rule_id": rule_id}


# ── 布控区域 CRUD ─────────────────────────────────────────────

class ZoneCreateRequest(BaseModel):
    name: str
    zone_type: str = "forbidden"  # "forbidden" | "danger"
    polygon: list[list[float]]  # [[x1,y1],[x2,y2],...]


def _read_camera_config() -> dict:
    return load_yaml(str(CONFIG_DIR / "camera_config.yaml")) or {}


def _save_camera_config(cfg: dict) -> None:
    with open(CONFIG_DIR / "camera_config.yaml", "w", encoding="utf-8") as f:
        _yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


def _hot_reload_zones() -> None:
    """保存后热更新引擎内的布控区域，无需重启。"""
    cfg = _read_camera_config()
    zones = cfg.get("zones", {})
    engine.behavior_reasoner.update_zones(
        forbidden_zones=zones.get("forbidden_zones", []),
        danger_zones=zones.get("danger_zones", []),
    )


@app.get("/api/v1/zones")
def get_zones():
    cfg = _read_camera_config()
    zones = cfg.get("zones", {})
    return {
        "forbidden_zones": zones.get("forbidden_zones", []),
        "danger_zones": zones.get("danger_zones", []),
    }


@app.post("/api/v1/zones")
def create_zone(request: ZoneCreateRequest):
    if request.zone_type not in ("forbidden", "danger"):
        raise HTTPException(status_code=400, detail="zone_type 必须是 forbidden 或 danger")
    if len(request.polygon) < 3:
        raise HTTPException(status_code=400, detail="多边形至少需要 3 个顶点")
    cfg = _read_camera_config()
    if "zones" not in cfg:
        cfg["zones"] = {}
    key = f"{request.zone_type}_zones"
    if key not in cfg["zones"]:
        cfg["zones"][key] = []
    zone_id = f"zone_{request.zone_type}_{int(time.time())}"
    cfg["zones"][key].append({"id": zone_id, "polygon": request.polygon})
    _save_camera_config(cfg)
    _hot_reload_zones()
    return {"status": "created", "zone_id": zone_id, "zone_type": request.zone_type, "polygon": request.polygon}


@app.delete("/api/v1/zones/{zone_id}")
def delete_zone(zone_id: str):
    cfg = _read_camera_config()
    zones = cfg.get("zones", {})
    for key in ("forbidden_zones", "danger_zones"):
        zone_list = zones.get(key, [])
        before = len(zone_list)
        zones[key] = [z for z in zone_list if z.get("id") != zone_id]
        if len(zones[key]) < before:
            cfg["zones"] = zones
            _save_camera_config(cfg)
            _hot_reload_zones()
            return {"status": "deleted", "zone_id": zone_id}
    raise HTTPException(status_code=404, detail=f"zone not found: {zone_id}")


@app.get("/api/v1/alerts")
def get_alerts(limit: int = 50):
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"items": read_alert_logs(results_dir, limit=limit)}


# 自动标注终审页静态服务（干净 URL，避免 media 接口查询参数被复制破坏）
from fastapi.staticfiles import StaticFiles
_annotation_dir = BASE_DIR / "results" / "auto_annotation"
_annotation_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/annotation", StaticFiles(directory=str(_annotation_dir)), name="annotation_static")


# 媒体白名单：目录限于项目根与 ../data（可用 CSA_MEDIA_ROOTS 追加），
# 扩展名限于视频/图片。防止 /api/v1/media?path=/etc/passwd 读取任意文件。
_MEDIA_ALLOWED_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v",
    ".jpg", ".jpeg", ".png", ".webp",
}


def _media_allowed_roots() -> tuple[Path, ...]:
    roots = [BASE_DIR, BASE_DIR.parent / "data"]
    extra = os.environ.get("CSA_MEDIA_ROOTS", "")
    roots.extend(Path(p) for p in extra.split(os.pathsep) if p.strip())
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(root.resolve())
        except Exception:
            continue
    return tuple(resolved)


def _media_path_allowed(path: Path) -> bool:
    if path.suffix.lower() not in _MEDIA_ALLOWED_SUFFIXES:
        return False
    try:
        resolved = path.resolve()  # 解析符号链接，防指向白名单外的链接
    except Exception:
        return False
    return any(resolved == root or resolved.is_relative_to(root)
               for root in _media_allowed_roots())


@app.get("/api/v1/media")
def get_media(path: str = Query(..., description="Absolute or project-relative local video path")):
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (BASE_DIR / candidate).resolve()

    # 先判权限再判存在性，避免用 403/404 差异探测文件是否存在
    if not _media_path_allowed(resolved):
        raise HTTPException(status_code=403, detail="media path not allowed")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="media file not found")

    media_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    return FileResponse(path=str(resolved), media_type=media_type, filename=resolved.name)


@app.post("/api/v1/evaluate")
def evaluate(request: EvaluateRequest):
    """启动后台评估任务（系统输出 vs VLM 伪真值）。返回 eval_id，用 GET 查询结果。"""
    if not request.ground_truth:
        raise HTTPException(status_code=400, detail="ground_truth JSON 路径必填（可用 tools/build_eval_ground_truth.py 生成）")
    video_path = Path(request.video_path)
    if not video_path.exists():
        candidate = BASE_DIR.parent / request.video_path
        if candidate.exists():
            video_path = candidate
        else:
            raise HTTPException(status_code=404, detail=f"video not found: {request.video_path}")
    gt_path = Path(request.ground_truth)
    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"ground_truth not found: {request.ground_truth}")

    eval_id = f"eval_{int(time.time() * 1000)}"
    EVAL_JOBS[eval_id] = {"status": "running", "progress": 0.0, "report": None, "error": ""}

    def job():
        def on_progress(payload: dict) -> None:
            if payload.get("stage") == "processing":
                EVAL_JOBS[eval_id]["progress"] = payload.get("progress", 0.0)
        try:
            from src.evaluation.evaluator import run_evaluation

            report = run_evaluation(
                video_path=str(video_path),
                ground_truth_path=str(gt_path),
                profile=request.profile,
                baseline=request.baseline,
                use_scene_context=request.use_scene_context,
                frame_stride=request.frame_stride,
                max_frames=request.max_frames,
                progress_cb=on_progress,
            )
            EVAL_JOBS[eval_id]["report"] = report
            EVAL_JOBS[eval_id]["status"] = "done"
            EVAL_JOBS[eval_id]["progress"] = 1.0
        except Exception as exc:
            EVAL_JOBS[eval_id]["status"] = "failed"
            EVAL_JOBS[eval_id]["error"] = str(exc)

    threading.Thread(target=job, daemon=True).start()
    return {"eval_id": eval_id, "status": "running", "endpoint": f"/api/v1/evaluate/{eval_id}"}


@app.get("/api/v1/evaluate/{eval_id}")
def get_evaluate(eval_id: str):
    job = EVAL_JOBS.get(eval_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"eval job not found: {eval_id}")
    payload = dict(job)
    if job.get("status") == "done" and job.get("report"):
        report = dict(job["report"])
        report.pop("records", None)  # 列表过大，详情看报告文件
        payload["report"] = report
    return payload


@app.post("/api/v1/train")
def train(request: TrainRequest):
    """启动后台 PoseClassifier 训练任务。返回 train_id，用 GET 查询进度。"""
    if TRAIN_JOBS.get("_active", {}).get("running"):
        raise HTTPException(status_code=409, detail="已有训练任务在运行")

    train_id = f"train_{int(time.time() * 1000)}"
    TRAIN_JOBS[train_id] = {"status": "running", "stage": "starting", "history": [],
                            "summary": None, "error": "", "request": request.model_dump()}
    TRAIN_JOBS["_active"] = {"running": True, "train_id": train_id}

    def job():
        try:
            from src.training.pose_classifier import train_pose_classifier

            def on_progress(payload: dict) -> None:
                state = TRAIN_JOBS[train_id]
                state["stage"] = payload.get("stage", "")
                if payload.get("stage") == "epoch":
                    state["history"].append({k: v for k, v in payload.items() if k != "stage"})
                elif payload.get("stage") == "done":
                    state["summary"] = payload

            summary = train_pose_classifier(
                data_dir=request.dataset_path if Path(request.dataset_path).is_absolute()
                else str(BASE_DIR / request.dataset_path),
                output_dir=str(BASE_DIR / "models" / "action_v2"),
                epochs=request.epochs,
                batch_size=request.batch_size,
                device=request.device,
                progress_cb=on_progress,
            )
            TRAIN_JOBS[train_id]["status"] = "done"
            TRAIN_JOBS[train_id]["summary"] = {
                "best_val_acc": summary["best_val_acc"],
                "onnx_path": summary["onnx_path"],
                "epochs": summary["epochs"],
            }
        except Exception as exc:
            TRAIN_JOBS[train_id]["status"] = "failed"
            TRAIN_JOBS[train_id]["error"] = str(exc)
        finally:
            TRAIN_JOBS["_active"] = {"running": False, "train_id": ""}

    threading.Thread(target=job, daemon=True).start()
    return {
        "status": "submitted",
        "train_id": train_id,
        "endpoint": f"/api/v1/train/{train_id}",
        "request": request.model_dump(),
    }


@app.get("/api/v1/train/{train_id}")
def get_train(train_id: str):
    job = TRAIN_JOBS.get(train_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"train job not found: {train_id}")
    return job


@app.post("/api/v1/vlm/review", response_model=VLMReviewResponse)
def vlm_review(request: VLMReviewRequest):
    """对单个候选事件做真实 VLM 复核（与推理流水线共用同一 VLMReviewer 实例）。"""
    event = request.candidate_event

    frames: list = []
    frame_b64 = event.get("frame_b64", "")
    if frame_b64:
        nparr = np.frombuffer(base64.b64decode(frame_b64), np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if decoded is not None:
            frames.append(decoded)
    track_crops = event.get("track_crops", [])
    for crop in track_crops:
        if isinstance(crop, (bytes, bytearray)):
            nparr = np.frombuffer(bytes(crop), np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded is not None:
                frames.append(decoded)
        elif isinstance(crop, np.ndarray):
            frames.append(crop)

    result = engine.vlm_reviewer.review_clip(frames=frames, track_id=event.get("track_id", ""))

    if result.get("status") not in {"success"}:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "VLM 复核不可用，未产生复核结论",
                "status": result.get("status", "error"),
                "reason": result.get("reasoning", ""),
            },
        )

    return VLMReviewResponse(
        event_id=event.get("event_id", f"evt_{request.frame_id}"),
        review_status="accepted",
        reviewed_behavior={
            "type": result.get("label", "Normal"),
            "confidence": float(result.get("confidence", 0.0)),
            "risk_level": event.get("risk_level", "LOW"),
        },
        role_scores=event.get("role_scores", {}),
        reason_tags=result.get("reason_tags", []),
        summary=result.get("reasoning", ""),
        latency_ms=int(result.get("latency_ms", 0)),
        model_name=engine.vlm_reviewer.model_name,
    )
