from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from src.core.action_recognizer import ActionRecognizer
from src.core.alert_adjudicator import AlertAdjudicator
from src.core.behavior_reasoner import BehaviorReasoner
from src.core.crop_classifier import CropClassifier, decide_crop_action
from src.core.detector import Detector
from src.core.interaction_analyzer import InteractionAnalyzer
from src.core.nl_rule_engine import NLAlertRuleEngine
from src.core.vlm_cache import VLMCropCache
from src.core.pose_estimator import PoseEstimator
from src.core.scene_narrator import SceneNarrator
from src.core.temporal_adjudicator import TemporalAdjudicator
from src.core.tracker import Tracker
from src.core.vlm_discovery import VLMDiscoveryScheduler
from src.core.vlm_reviewer import VLMReviewer, build_forced_prompt
from src.pipeline.alert_system import AlertSystem
from src.pipeline.clip_exporter import ClipExporter, ClipRequest
from src.pipeline.frame_buffer import FrameRingBuffer
from src.pipeline.vlm_router import VLMRouter
from src.utils.schemas import Behavior, Detection, FrameResult, VLMReview


RISK_MAP = {
    "Normal": "LOW",
    "H01": "HIGH",
    "H02": "HIGH",
    "H03": "HIGH",
    "H04": "HIGH",
    "H05": "HIGH",
    "M01": "MEDIUM",
    "M02": "MEDIUM",
    "M03": "MEDIUM",
    "S01": "MEDIUM",
    "S02": "MEDIUM",
}


class InferenceEngine:
    @staticmethod
    def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
        ax1, ay1, aw, ah = [float(v) for v in a]
        bx1, by1, bw, bh = [float(v) for v in b]
        ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
        bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def __init__(self, model_cfg: dict, vlm_cfg: dict, output_dir: str = "results"):
        runtime_cfg = dict(model_cfg.get("runtime", {}))
        detector_cfg = dict(model_cfg.get("detector", {}))
        detector_cfg["runtime"] = runtime_cfg
        pose_cfg = dict(model_cfg.get("pose", {}))
        pose_cfg["runtime"] = runtime_cfg
        action_cfg = dict(model_cfg.get("action", {}))
        # 仅 action 模型走 CPU 推理，规避 cuDNN 在 CUDA 13.0 下的 Conv+bias 兼容性问题
        action_runtime = dict(runtime_cfg)
        action_runtime["execution_provider"] = "CPUExecutionProvider"
        action_cfg["runtime"] = action_runtime

        self.detector = Detector(detector_cfg)
        self.tracker = Tracker(model_cfg.get("tracker", {}))
        self.pose_estimator = PoseEstimator(pose_cfg)
        self.action_recognizer = ActionRecognizer(action_cfg)
        inference_cfg = model_cfg.get("inference", {})
        self.pose_stride = int(max(1, inference_cfg.get("pose_stride", 8)))
        self.max_pose_tracks = int(max(1, inference_cfg.get("max_pose_tracks", 2)))
        self.action_stride = int(max(1, inference_cfg.get("action_stride", 6)))
        self.max_action_tracks = int(max(1, inference_cfg.get("max_action_tracks", 1)))
        self.action_min_track_score = float(max(0.0, inference_cfg.get("action_min_track_score", 0.0)))
        self.action_warmup_frames = int(max(0, inference_cfg.get("action_warmup_frames", 0)))
        self.action_min_track_age = int(max(1, inference_cfg.get("action_min_track_age", 1)))
        self.action_default_label = str(inference_cfg.get("action_default_label", "Normal"))
        self.action_default_confidence = float(min(1.0, max(0.0, inference_cfg.get("action_default_confidence", 0.60))))
        self.pose_motion_gate_iou_threshold = float(inference_cfg.get("pose_motion_gate_iou_threshold", 1.01))
        self.enable_pose_motion_gate = bool(inference_cfg.get("enable_pose_motion_gate", False))
        self.pose_force_refresh_gap = int(max(1, inference_cfg.get("pose_force_refresh_gap", max(1, self.pose_stride * 2))))
        self.enable_detector_gating = bool(inference_cfg.get("enable_detector_gating", False))
        self.detector_stride = int(max(1, inference_cfg.get("detector_stride", 1)))
        self.detector_max_skip_frames = int(max(self.detector_stride, inference_cfg.get("detector_max_skip_frames", self.detector_stride * 2)))
        self.detector_motion_threshold = float(max(0.0, inference_cfg.get("detector_motion_threshold", 0.012)))
        self.detector_low_motion_threshold = float(max(0.0, inference_cfg.get("detector_low_motion_threshold", self.detector_motion_threshold * 0.5)))
        self.detector_warmup_frames = int(max(0, inference_cfg.get("detector_warmup_frames", 1)))
        self.behavior_reasoner = BehaviorReasoner(model_cfg.get("behavior_heuristics", {}))
        self.temporal_adjudicator = TemporalAdjudicator(model_cfg.get("temporal_adjudicator", {}))
        self.pose_cache: dict[str, list[list[float]]] = {}
        self.pose_last_bbox: dict[str, list[float]] = {}
        self.pose_last_frame: dict[str, int] = {}
        self.action_cache: dict[str, tuple[str, float]] = {}
        self.track_seen_frames: dict[str, int] = {}
        self.interaction_analyzer = InteractionAnalyzer()
        self.vlm_reviewer = VLMReviewer(vlm_cfg)
        self.vlm_router = VLMRouter(vlm_cfg)
        alert_policy_cfg = dict(model_cfg.get("alert_policy", {})) if isinstance(model_cfg, dict) else {}
        if not alert_policy_cfg:
            alert_policy_cfg = {
                "min_consecutive_frames": inference_cfg.get("min_consecutive_frames", 3),
                "cooldown_seconds": inference_cfg.get("cooldown_seconds", 5),
                "high_risk_min_confidence": inference_cfg.get("high_risk_min_confidence", 0.55),
            }
        self.alert_system = AlertSystem(output_dir, config=alert_policy_cfg)
        # 告警证据切片（docs/06 §1）：异步队列，默认只对本地文件源生效
        self.clip_exporter = ClipExporter(
            output_dir, dict((model_cfg or {}).get("clip_export", {})))
        self._task_id = ""
        self._source_url = ""
        self._last_detector_gray: np.ndarray | None = None
        self._detector_skip_count = 0

        # ── B2: ORT 模型 warmup（消除首帧 5-10x 慢启动） ──
        self._warmup_models()

        # ── C1: ID 切换检测（每个 track 记录上一帧 bbox，跳变时清零缓存） ──
        self._track_prev_bbox: dict[str, list[float]] = {}
        self._id_switch_distance_ratio = 2.0  # 质心位移/bbox对角线 > 此值判定为切换
        self._id_switch_area_ratio = 3.0     # bbox面积突变 > 此值判定为切换

        # ── B1: VLM 异步复核线程池 + 缓存锁 ──
        self._vlm_async_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-async")
        self._vlm_cache = VLMCropCache(max_size=100, ttl_seconds=30.0)
        self._cache_lock = threading.Lock()

        # ── VLM 发现器（第一期）：事件审查 + 周期采样；异步、预算受限、不覆盖主判定 ──
        self.vlm_discovery = VLMDiscoveryScheduler(
            dict((vlm_cfg or {}).get("discovery", {})),
            reviewer=self.vlm_reviewer,
            cache_lookup=self._vlm_cache.get,
        )

        # 2D crop 分类器（M01/M02/H03 本地确认钥匙）：默认 CPU，避免与检测抢 GPU
        self.crop_classifier = CropClassifier(dict((model_cfg or {}).get("crop_classifier", {})))
        self._crop_review_cache: dict[str, tuple[int, str, float]] = {}
        # M01 提案通道状态：连续命中次数（crop 主提案，规则只做佐证）
        self._crop_proposal_hits: dict[str, int] = {}

        # ── B6: 三阶段流水线（detector/tracker → pose → action/reasoner/post） ──
        _pipeline_cfg = dict((model_cfg or {}).get("inference", {}).get("pipeline", {}))
        self._pipeline_enabled = bool(_pipeline_cfg.get("enabled", False))
        self._pipe_queue_size = max(2, int(_pipeline_cfg.get("queue_size", 4)))
        self._pipe_generation = 0
        self._pipe_dropped = 0
        self._pipe_results: deque = deque()
        self._pipe_results_lock = threading.Lock()
        self._last_pipeline_error = ""
        if self._pipeline_enabled:
            from queue import Queue
            self._pipe_q1 = Queue(maxsize=self._pipe_queue_size)
            self._pipe_q2 = Queue(maxsize=self._pipe_queue_size)
            self._pipe_q3 = Queue(maxsize=self._pipe_queue_size)
            for _stage_no in (1, 2, 3):
                threading.Thread(
                    target=self._pipeline_worker, args=(_stage_no,),
                    daemon=True, name=f"pipe-stage{_stage_no}",
                ).start()

        # ── 场景理解与告警裁决（均为异步旁路，不阻塞推理流水线） ──
        vlm_cfg = vlm_cfg if isinstance(vlm_cfg, dict) else {}
        fb_cfg = dict(vlm_cfg.get("frame_buffer", {}))
        self.frame_buffer = FrameRingBuffer(
            max_seconds=float(fb_cfg.get("max_seconds", 30.0)),
            samples_per_second=float(fb_cfg.get("samples_per_second", 2.0)),
            target_width=int(fb_cfg.get("target_width", 640)),
        )
        self.scene_narrator = SceneNarrator(vlm_cfg, output_dir=output_dir)
        self.scene_narrator.attach_vlm_router(self.vlm_router)  # B3
        self.scene_narrator.attach_buffer(self.frame_buffer)
        self.alert_adjudicator = AlertAdjudicator(vlm_cfg, output_dir=output_dir)
        self.alert_adjudicator.attach_vlm_router(self.vlm_router)  # B3
        self.alert_adjudicator.attach_buffer(self.frame_buffer)
        self.alert_adjudicator.attach_narrator(self.scene_narrator)
        self.nl_rule_engine = NLAlertRuleEngine(vlm_cfg, self.frame_buffer, output_dir=output_dir)
        # ROADMAP Stage 1/2：新行为备忘 + 行为原型库（旁路，事件级触发；缓冲在下方创建后再 attach）
        from src.core.novel_behavior_memo import NovelBehaviorMemo
        from src.core.behavior_prototypes import BehaviorPrototypes
        self.novel_behavior_memo = NovelBehaviorMemo(vlm_cfg, output_dir=output_dir)
        # 原型是跨任务累积的全局知识，固定存 results/prototypes/，不随任务 output_dir 隔离
        self.behavior_prototypes = BehaviorPrototypes(vlm_cfg, output_dir="results")
        self.novel_behavior_memo.attach_prototypes(self.behavior_prototypes)
        self.nl_rule_engine.attach_vlm_router(self.vlm_router)  # B3
        self._last_frame_id = 0
        self.track_label_history: dict[str, deque] = {}
        # 诊断计数：裁决队列满被拒的告警数（WEAKNESS_AUDIT #1 修复后仍可能有界丢弃）
        self._adjudicate_submit_dropped = 0
        # 强制 VLM 复核（如 H03 跌倒，与坐姿/半躺易混淆）：越过三级决策直接复核，逐目标节流
        fvr_cfg = dict(vlm_cfg.get("force_vlm_review", {}))
        self.force_vlm_classes = {str(c) for c in fvr_cfg.get("classes", []) if str(c)}
        self.force_vlm_throttle_frames = int(max(1, fvr_cfg.get("throttle_frames", 25)))
        # 复核结论的有效期：VLM 一次往返常需 3–10s（36–120 帧），用 throttle(75) 判定缓存
        # 会把迟到的否决当"过期"丢弃，导致强几何 H03 的误报永远无法被纠正。
        self._veto_cache_frames = int(max(self.force_vlm_throttle_frames,
                                         fvr_cfg.get("veto_cache_frames", 300)))
        self.geom_strong_parallel_review = bool(fvr_cfg.get("geom_strong_parallel_review", True))
        self._geom_strong_reviews = 0
        # 强几何 H03 的"两次同向否决才压制"状态：缓存结论可能抖动，
        # 单次 VLM 误判 Normal 就压制 15s 会漏掉真实跌倒。
        self._geom_veto_last_cache: dict[str, int] = {}
        self._geom_veto_streak: dict[str, int] = {}
        # 全局复核预算：VLM 与 ONNX 共用 GPU，按 track/标签的节流挡不住"多 track/多标签"的叠加，
        # 这里再加一层全局令牌桶（默认 8 次/分钟），保证检测推理始终有 GPU 可用。
        self._review_rate_per_min = float(max(0.0, fvr_cfg.get("max_per_minute", 8)))
        self._review_tokens = 4.0
        self._review_token_ts = time.monotonic()
        self._review_rate_limited = 0
        self.track_vlm_review_cache: dict[str, tuple[int, str, float]] = {}
        # C2: 时序标签平滑参数
        self._smooth_window = 5
        # A: 检测事实注入——近窗异常寄存器（track → 最近一次异常判定），叙述注入按窗口聚合，抗标签瞬时波动
        self._recent_abnormal: dict[str, dict] = {}
        # 归因诊断：action_cache 中由 VLM 异步回写产生的 track（用于区分 cache 沿用的来源）
        self._vlm_cache_sourced: set[str] = set()
        # 归因修复：VLM 否决锁存（track_id -> {label: until_frame}），规则被复核否决后压制约15s
        self._vlm_veto: dict[str, dict[str, int]] = {}
        self._vlm_veto_seconds = 15.0
        self._source_fps = 25.0
        # 双钥匙 A：M01 的 VLM 确认寄存器（track -> until_frame），与规则候选缺一不可
        self._vlm_confirmed: dict[str, int] = {}
        self._vlm_confirm_hold_seconds = 12.0
        # 问题1：原帧复核通道——环形缓冲存 960 降采样帧（叙述/裁决用），
        # 本缓冲直存原始分辨率帧（person-crop 复核专用），真实细节翻倍。
        # 复核只需取近帧裁剪，独立收口（默认 48 帧 ≈ 300MB@1080p）：
        # 旧实现继承 30s×10fps=300 帧，1080p 下常驻约 1.9GB（WEAKNESS_AUDIT #9）
        _fbo_cfg = dict(vlm_cfg.get("frame_buffer_orig", {})) if isinstance(vlm_cfg, dict) else {}
        self.frame_buffer_orig = FrameRingBuffer(
            max_seconds=float(_fbo_cfg.get("max_seconds", 12.0)),
            samples_per_second=float(_fbo_cfg.get("samples_per_second", 2.0)),
            target_width=0,
            max_frames=int(_fbo_cfg.get("max_frames", 48)),
        )
        # Stage 1/2 缓冲接线（frame_buffer_orig 创建后）
        self.novel_behavior_memo.attach_vlm_router(self.vlm_router)
        self.novel_behavior_memo.attach_buffer(self.frame_buffer)
        self.novel_behavior_memo.attach_buffer_orig(self.frame_buffer_orig)
        self.novel_behavior_memo.attach_pose_buffers(self.action_recognizer.track_pose_buffers)
        # 复核判定平滑：同一 track 最近 3 次复核多数投票，防单次抖动翻转确认/否决状态
        self._review_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))
        self._veto_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=2))
        # 复核在飞名额（track|label）：VLM 吞吐有限，无上限会把队列压成"永远处理早帧"
        self._review_inflight: set[tuple[str, str]] = set()
        # VLM 与 ONNX 共用 GPU：并发复核过多会拖慢检测，导致前端框滞后。
        # 2 个并发足够覆盖灰区/规则复核，又能把 GPU 争用压在可接受范围。
        self._review_inflight_max = 2
        # 确认/否决的空间锚定：ID 切换频繁场景（P007→P009→P008 连环重编号）下，
        # track_id 键的确认窗口随 ID 丢失——附加质心网格键（80px）作空间备份
        self._vlm_confirmed_spatial: dict[tuple, int] = {}
        self._vlm_veto_spatial: dict[tuple, dict[str, int]] = {}
        self._spatial_grid = 80.0
        self._smooth_min_votes = 3
        self.track_smooth_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._smooth_window))
        # 消融开关：关闭后灰区复核不注入场景上下文（评估用）
        self.review_context_enabled = True

    def begin_task(self, task_id: str) -> None:
        """新任务开始：重置所有按帧号调度的旁路模块。

        帧号在任务间归零重来，环形缓冲/叙述调度/裁决等待/规则引擎调度
        若跨任务残留，会导致新任务前半段无叙述、开放规则不触发等问题。
        """
        self._task_id = str(task_id or "")
        self.clip_exporter.begin_task(task_id)
        self.frame_buffer.clear()
        self.frame_buffer_orig.clear()
        self.scene_narrator.begin_task(task_id)
        self.alert_adjudicator.reset_for_task()
        self.nl_rule_engine.reset_scheduling()
        self.novel_behavior_memo.begin_task(task_id)
        # C2/B5 状态同样按任务隔离：上一任务的平滑历史与 crop 缓存不得泄漏
        self.track_smooth_history.clear()
        self._vlm_cache.clear()
        # 逐 track 标签/复核缓存也必须清：track_id 在新任务里从 P001 重新编号，
        # 上一任务的 (H03, 0.72) 会直接套到新任务第一个目标上（实测 smoking 任务
        # 第 1 帧就把 P003 标成 H03 并误告警）
        self.action_cache.clear()
        self.track_vlm_review_cache.clear()
        self.pose_cache.clear()
        self.track_label_history.clear()
        self._review_inflight.clear()
        # 规则计数器（fall/h03/smoke…）同样按 track_id 存，跨任务会直接满足"连续帧"门槛
        self.behavior_reasoner.clear_stale_tracks(set())
        # 告警连续帧计数与冷却同理：跨任务残留会让新任务第 1 帧直接满足"连续 3 帧"
        self.alert_system._consecutive_frames.clear()
        self.alert_system._last_alert_time.clear()
        self._adjudicate_submit_dropped = 0
        self._recent_abnormal.clear()
        self._vlm_cache_sourced.clear()
        self._vlm_veto.clear()
        self._vlm_confirmed.clear()
        self._review_history.clear()
        self._veto_history.clear()
        self._vlm_confirmed_spatial.clear()
        self._vlm_veto_spatial.clear()
        # B6: 管道状态隔离——代际 +1 使各阶段丢弃旧任务残留帧，并清空队列/结果
        self._pipe_generation += 1
        for _q_name in ("_pipe_q1", "_pipe_q2", "_pipe_q3"):
            _q = getattr(self, _q_name, None)
            if _q is not None:
                try:
                    while True:
                        _q.get_nowait()
                except Exception:
                    pass
        with self._pipe_results_lock:
            self._pipe_results.clear()

    def set_source_url(self, url: str) -> None:
        """由任务层注入视频源地址（切片路径 A 需要原文件路径）。"""
        self._source_url = str(url or "")

    def _submit_clip(self, alert: dict, frame_id: int, event_filename: str) -> None:
        """告警证据切片：异步提交，任何失败都不得影响告警主链路。"""
        try:
            if not self.clip_exporter.enabled or not self._source_url:
                return
            self.clip_exporter.submit(ClipRequest(
                task_id=self._task_id,
                frame_id=int(frame_id),
                event_filename=event_filename,
                source_url=self._source_url,
                video_time_sec=float(frame_id) / max(1.0, float(self._source_fps)),
                before_sec=self.clip_exporter.before_sec,
                after_sec=self.clip_exporter.after_sec,
            ))
        except Exception:
            pass

    def set_source_fps(self, fps: float) -> None:
        """由任务层注入视频源帧率，驱动叙述/裁决的视频时间计算。"""
        self._source_fps = float(fps if fps and fps > 1.0 else 25.0)
        self.frame_buffer.set_source_fps(fps)
        self.scene_narrator.set_source_fps(fps)
        self.alert_adjudicator.set_source_fps(fps)
        self.behavior_reasoner.set_source_fps(fps)
        self.nl_rule_engine.set_source_fps(fps)
        self.novel_behavior_memo.set_source_fps(fps)

    def close(self) -> None:
        self.scene_narrator.stop()
        self.alert_adjudicator.stop()
        self.nl_rule_engine.stop()
        self.novel_behavior_memo.stop()
        self._vlm_async_executor.shutdown(wait=False, cancel_futures=True)

    # ── B2: ORT warmup ───────────────────────────────────────
    def _warmup_models(self) -> None:
        """启动时各模型跑一次 dummy 推理，触发 ONNX graph 优化与内存分配预热。"""
        try:
            dummy = np.zeros((360, 640, 3), dtype=np.uint8)
            if self.detector.is_real_model_loaded:
                self.detector.infer(dummy)
            if self.pose_estimator.is_real_model_loaded:
                fake_track = {"track_id": "warmup", "bbox": [100, 50, 200, 250], "score": 0.9}
                self.pose_estimator.infer_batch(dummy, [fake_track])
            # PoseClassifier warmup：填充 32 帧零姿态
            if self.action_recognizer.pose_classifier_loaded:
                zero_pose = np.zeros((133, 3), dtype=np.float32)
                for _ in range(self.action_recognizer.input_clip_len):
                    self.action_recognizer.append_pose("warmup", zero_pose.tolist())
                self.action_recognizer.infer_pose_classifier_batch(["warmup"])
                self.action_recognizer.clear_stale_tracks(set())  # 清理 warmup track
        except Exception as exc:
            print(f"[warmup] skipped: {exc}")

    # ── C1: ID 切换检测 + 缓冲清零 ──────────────────────────
    def _clear_track_caches(self, track_id: str) -> None:
        """ID 切换时清零该 track 的所有逐目标缓存，防止混人序列污染。"""
        self.action_recognizer.clear_stale_tracks({track_id})
        self.behavior_reasoner.clear_stale_tracks({track_id})
        with self._cache_lock:
            self.track_vlm_review_cache.pop(track_id, None)
            self.track_label_history.pop(track_id, None)

    def _detect_and_handle_id_switches(self, tracks: list[dict]) -> None:
        """检测 bbox 跳变（ID 切换），清零被污染的逐目标缓存。"""
        for track in tracks:
            track_id = str(track["track_id"])
            bbox = track.get("bbox", [0, 0, 0, 0])
            prev_bbox = self._track_prev_bbox.get(track_id)
            if prev_bbox is not None and len(prev_bbox) == 4 and len(bbox) == 4:
                # 质心位移
                cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                px, py = prev_bbox[0] + prev_bbox[2] / 2, prev_bbox[1] + prev_bbox[3] / 2
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                diag = max(1.0, (bbox[2] ** 2 + bbox[3] ** 2) ** 0.5)
                ratio = dist / diag
                # 面积突变
                area = max(1.0, bbox[2] * bbox[3])
                prev_area = max(1.0, prev_bbox[2] * prev_bbox[3])
                area_ratio = max(area / prev_area, prev_area / area)
                if ratio > self._id_switch_distance_ratio or area_ratio > self._id_switch_area_ratio:
                    self._clear_track_caches(track_id)
            self._track_prev_bbox[track_id] = list(bbox)
        # 清理消失的 track + 确认窗口继承（同位置新 ID 延续 VLM 确认，防 ID 切换丢钥匙）
        active_ids = {str(t["track_id"]) for t in tracks}
        stale = [tid for tid in self._track_prev_bbox if tid not in active_ids]
        for tid in stale:
            self._track_prev_bbox.pop(tid, None)
        for tid in stale:
            moved = {k: v for k, v in list(self._vlm_confirmed.items())
                     if k == tid or k.startswith(f"{tid}|")}
            if not moved:
                continue
            for k in moved:
                self._vlm_confirmed.pop(k, None)
            # 找与消失 track 上一帧位置最近的现存 track，继承确认窗口（按类别迁移）
            best_tid, best_d = None, float("inf")
            for t in tracks:
                cand = str(t["track_id"])
                if cand == tid:
                    continue
                bx, by, bw, bh = [float(v) for v in t.get("bbox", [0, 0, 0, 0])]
                d = (bx + bw / 2) ** 2 + (by + bh / 2) ** 2
                if d < best_d:
                    best_d, best_tid = d, cand
            if best_tid is not None:
                for k, v in moved.items():
                    label = k.split("|", 1)[1] if "|" in k else "M01"
                    nk = self._confirm_key(best_tid, label)
                    self._vlm_confirmed[nk] = max(v, self._vlm_confirmed.get(nk, 0))

    # ── B1: VLM 异步复核 ─────────────────────────────────────
    def _spatial_key(self, bbox: list[float]) -> tuple:
        x, y, w, h = [float(v) for v in bbox]
        return (int((x + w / 2) // self._spatial_grid), int((y + h / 2) // self._spatial_grid))

    def _confirm_key(self, track_id: str, label: str) -> str:
        """确认窗口按 (track, 类别) 存储：M01 的确认不能给 H03 当钥匙。"""
        return f"{track_id}|{label}"

    def _select_review_frames_by_hand_mouth(self, track_id: str, bbox: list[float], k: int = 3,
                                              buffer=None):
        """复核主动选帧：按手-嘴距离排序环形缓冲帧，取距离最小（动作峰值）的 k 帧。

        吸烟是间歇动作，均匀采样常错过"手举烟"瞬间；姿态几何提供廉价的峰值信号。
        """
        entries = (buffer or self.frame_buffer).recent_entries(12)  # 最近 ~12 个缓冲帧
        if not entries:
            return []
        scored = []
        for fid, full_f in entries:
            kps = self.pose_cache.get(track_id)
            score = 0.0
            try:
                if kps:
                    score = -self.behavior_reasoner.hand_mouth_distance(kps, bbox)
            except Exception:
                score = 0.0
            scored.append((score, fid, full_f))
        scored.sort(key=lambda t: t[0])  # 距离最小在前
        return [(fid, f) for _, fid, f in scored[:k]]

    @staticmethod
    def _frame_dhash(frame: np.ndarray) -> int:
        """8x8 差分感知哈希，用于发现器去重（同一画面短时间内不重复送审）。"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            bits = 0
            for value in (small[:, 1:] > small[:, :-1]).flatten():
                bits = (bits << 1) | int(bool(value))
            return bits
        except Exception:
            return 0

    def _build_discovery_frames(self, track: dict, frame: np.ndarray) -> list[np.ndarray]:
        """发现器输入：一张带红框的全景 + 最近 2 张历史帧（场景级视野，避免只看 crop）。"""
        frames: list[np.ndarray] = []
        if frame is not None:
            pano = frame
            h, w = pano.shape[:2]
            if max(h, w) > 720:
                scale = 720.0 / max(h, w)
                pano = cv2.resize(pano, (int(w * scale), int(h * scale)))
            pano = pano.copy()
            bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
            sx, sy = pano.shape[1] / max(1.0, float(w)), pano.shape[0] / max(1.0, float(h))
            x1, y1 = int(bbox[0] * sx), int(bbox[1] * sy)
            x2, y2 = int((bbox[0] + bbox[2]) * sx), int((bbox[1] + bbox[3]) * sy)
            cv2.rectangle(pano, (x1, y1), (x2, y2), (0, 0, 255), 2)
            frames.append(pano)
        for _, hist_frame in self.frame_buffer.sample(2):
            if hist_frame is not None and hist_frame is not frame:
                frames.append(hist_frame)
        return frames[:3]

    def _build_crop_discovery_frames(self, track: dict, frame: np.ndarray,
                                     max_px: int = 1024) -> list[np.ndarray]:
        """M01 专用输入：仅人物裁剪（外扩 15%），当前帧 + 最近缓冲帧。

        实测：全景会稀释小目标注意力（pano+crop 四种分辨率组合全 0/5），
        crop-only + 定向 prompt 才能命中（3/5；两帧 5/5）。
        """
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = [int(round(v)) for v in bbox]
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        frame_h, frame_w = frame.shape[:2]
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(frame_w, x + w + pad_x), min(frame_h, y + h + pad_y)

        def _prepare(img: np.ndarray) -> np.ndarray | None:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            ch, cw = crop.shape[:2]
            scale = float(max_px) / max(1, ch, cw)
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            return cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                              interpolation=interp)

        frames: list[np.ndarray] = []
        current = _prepare(frame)
        if current is not None:
            frames.append(current)
        entries = self.frame_buffer.recent_entries(6)
        if len(entries) > 1:
            prev = _prepare(entries[0][1])
            if prev is not None:
                frames.append(prev)
        return frames[:2]

    # 向后兼容旧命名
    _build_m01_discovery_frames = _build_crop_discovery_frames

    @staticmethod
    def _crop_resize(frame: np.ndarray, bbox: list[float], max_px: int,
                     pad_ratio: float = 0.15) -> np.ndarray | None:
        x, y, w, h = [int(round(float(v))) for v in bbox]
        pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        scale = float(max_px) / max(1, ch, cw)
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                          interpolation=interp)

    @staticmethod
    def _peak_frame(entries: list) -> np.ndarray | None:
        """在最近缓冲帧里挑帧间差异最大的一帧（运动峰值），而不是固定取上一帧。"""
        if len(entries) < 2:
            return None
        best = None
        best_score = -1.0
        prev_gray = None
        for _, img in entries:
            try:
                small = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            except Exception:
                continue
            if prev_gray is not None:
                score = float(np.mean(cv2.absdiff(gray, prev_gray)))
                if score > best_score:
                    best_score = score
                    best = img
            prev_gray = gray
        return best

    def _build_h_single_frames(self, track: dict, frame: np.ndarray,
                               max_px: int = 1024) -> list[np.ndarray]:
        """H03/H04/H05：全景（红框）+ 人物裁剪 + 上一帧全景（场景变化参照）。"""
        frames: list[np.ndarray] = []
        h, w = frame.shape[:2]
        scale = min(1.0, 960.0 / max(1, h, w))
        pano = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame.copy()
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        cv2.rectangle(pano, (int(bbox[0] * scale), int(bbox[1] * scale)),
                      (int((bbox[0] + bbox[2]) * scale), int((bbox[1] + bbox[3]) * scale)),
                      (0, 0, 255), 2)
        frames.append(pano)
        crop = self._crop_resize(frame, bbox, max_px)
        if crop is not None:
            frames.append(crop)
        entries = self.frame_buffer.recent_entries(8)
        peak = self._peak_frame(entries)
        if peak is not None:
            ph, pw = peak.shape[:2]
            ps = min(1.0, 720.0 / max(1, ph, pw))
            frames.append(cv2.resize(peak, (int(pw * ps), int(ph * ps))) if ps < 1.0 else peak)
        return frames[:3]

    def _build_h_pair_frames(self, track: dict, peer_bbox: list[float] | None,
                             frame: np.ndarray, max_px: int = 1024) -> list[np.ndarray]:
        """H01/H02：全景（双框）+ 两名当事人裁剪。"""
        frames: list[np.ndarray] = []
        h, w = frame.shape[:2]
        scale = min(1.0, 960.0 / max(1, h, w))
        pano = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame.copy()
        boxes = [track.get("bbox", [0, 0, 0, 0])]
        if peer_bbox is not None:
            boxes.append(peer_bbox)
        for idx, box in enumerate(boxes):
            b = [float(v) for v in box]
            color = (0, 0, 255) if idx == 0 else (0, 200, 0)
            cv2.rectangle(pano, (int(b[0] * scale), int(b[1] * scale)),
                          (int((b[0] + b[2]) * scale), int((b[1] + b[3]) * scale)), color, 2)
        frames.append(pano)
        crop = self._crop_resize(frame, [float(v) for v in track.get("bbox", [0, 0, 0, 0])], max_px)
        if crop is not None:
            frames.append(crop)
        if peer_bbox is not None:
            peer_crop = self._crop_resize(frame, [float(v) for v in peer_bbox], max_px)
            if peer_crop is not None:
                frames.append(peer_crop)
        return frames[:3]

    def _maybe_submit_discovery(self, track: dict, frame: np.ndarray, frame_id: int,
                                label: str, confidence: float, source: str,
                                tracks: list[dict] | None = None) -> None:
        """发现器入口：策略判断（预算/冷却/在飞）→ 后台异步送审，主循环不做等待。"""
        scheduler = getattr(self, "vlm_discovery", None)
        if scheduler is None or not scheduler.enabled:
            return
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = [int(round(v)) for v in bbox]
        h_f, w_f = frame.shape[:2]
        crop = frame[max(0, y):min(h_f, y + h), max(0, x):min(w_f, x + w)]
        if crop.size == 0:
            return
        label_text = str(label)
        track_id = str(track.get("track_id", ""))
        peer_id = ""
        peer_bbox: list[float] | None = None
        channel = "generic"
        if label_text == "M01":
            channel = "m01"
        elif label_text == "M02":
            channel = "m02"
        elif label_text in ("H01", "H02"):
            # 双人事件：按 pair 聚合提案（同一对只占一个冷却槽），并送双方裁剪
            if tracks:
                try:
                    close = self.behavior_reasoner._close_peer_ids(
                        track_id, bbox, tracks)
                except Exception:
                    close = set()
                for peer in tracks:
                    pid = str(peer.get("track_id", ""))
                    if pid and pid != track_id and pid in close:
                        peer_id = pid
                        peer_bbox = [float(v) for v in peer.get("bbox", [0, 0, 0, 0])]
                        break
            channel = "h_pair" if peer_bbox is not None else "h_single"
        elif label_text in ("H03", "H04", "H05"):
            channel = "h_single"

        proposal_track_id = track_id
        if channel == "h_pair" and peer_id:
            try:
                proposal_track_id = self.behavior_reasoner._pair_key(track_id, peer_id)
            except Exception:
                proposal_track_id = f"{track_id}:{peer_id}"

        proposal = scheduler.maybe_propose(
            track_id=proposal_track_id,
            frame_id=int(frame_id),
            label=str(label),
            confidence=float(confidence),
            source=str(source),
            track_score=float(track.get("score", 0.0)),
            video_time=float(frame_id) / max(1e-6, float(self._source_fps)),
            image_hash=self._frame_dhash(crop),
            channel=channel,
        )
        if proposal is None:
            return
        chan_cfg = (getattr(scheduler, "channels", {}) or {}).get(channel, {})
        image_max_px = int(chan_cfg.get("image_max_px", 1024))
        if channel in ("m01", "m02"):
            frames = self._build_crop_discovery_frames(track, frame, image_max_px)
        elif channel == "h_pair":
            frames = self._build_h_pair_frames(track, peer_bbox, frame, image_max_px)
        elif channel == "h_single":
            frames = self._build_h_single_frames(track, frame, image_max_px)
        else:
            frames = self._build_discovery_frames(track, frame)
        if not frames or not scheduler.submit(proposal, frames):
            scheduler.cancel(proposal)

    def _crop_verdict(self, track: dict, frame: np.ndarray, frame_id: int) -> tuple[str, float]:
        """按 track 节流运行 crop 分类器，返回 (label, confidence)；不可用时返回 ("", 0)。"""
        classifier = getattr(self, "crop_classifier", None)
        if classifier is None or not getattr(classifier, "loaded", False):
            return "", 0.0
        track_id = str(track.get("track_id", ""))
        cached = self._crop_review_cache.get(track_id)
        if cached is not None and (frame_id - int(cached[0])) < classifier.throttle_frames:
            return cached[1], cached[2]

        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = [int(round(v)) for v in bbox]
        frame_h, frame_w = frame.shape[:2]
        crop = frame[max(0, y):min(frame_h, y + h), max(0, x):min(frame_w, x + w)]
        result = classifier.predict(crop)
        pred_label = str(result.get("label", ""))
        pred_conf = float(result.get("confidence", 0.0))
        self._crop_review_cache[track_id] = (int(frame_id), pred_label, pred_conf)
        return pred_label, pred_conf

    def _maybe_crop_proposal(self, track: dict, frame: np.ndarray, frame_id: int,
                             label: str, confidence: float, src_tag: str,
                             base_label: str, base_confidence: float) -> tuple[str, float, str, bool]:
        """crop 主提案通道：连续 N 次判 M01 且过阈值时，在 Normal 帧上提案 M01。

        背景：M01 的 pose 近嘴规则在办公场景误报高（454 帧/900 帧）、吸烟场景覆盖不全；
        实测 crop 分类器对 M01 的区分度更好（吸烟 96 vs 办公 16）。因此改为 crop 主提案，
        pose 规则只做佐证（`m01_require_crop` 门控）。
        """
        classifier = getattr(self, "crop_classifier", None)
        if classifier is None or not getattr(classifier, "loaded", False):
            return label, confidence, src_tag, False
        if not classifier.propose_m01:
            return label, confidence, src_tag, False

        track_id = str(track.get("track_id", ""))
        if label != "Normal":
            self._crop_proposal_hits[track_id] = 0
            return label, confidence, src_tag, False

        pred_label, pred_conf = self._crop_verdict(track, frame, frame_id)
        if pred_label == "M01" and pred_conf >= classifier.proposal_conf:
            hits = int(self._crop_proposal_hits.get(track_id, 0)) + 1
        else:
            hits = 0
        self._crop_proposal_hits[track_id] = hits
        if hits >= classifier.proposal_hits:
            return "M01", max(float(confidence), float(pred_conf)), "crop_proposal", True
        return label, confidence, src_tag, False

    def _apply_crop_review(self, track: dict, frame: np.ndarray, frame_id: int,
                           label: str, confidence: float, src_tag: str,
                           base_label: str, base_confidence: float) -> tuple[str, float, str]:
        """本地 crop 分类器：M01 规则门控 + M01/M02/H03 确认/否决（按 track 节流缓存）。"""
        classifier = getattr(self, "crop_classifier", None)
        if classifier is None or not getattr(classifier, "loaded", False):
            return label, confidence, src_tag
        if label not in classifier.review_classes:
            return label, confidence, src_tag

        pred_label, pred_conf = self._crop_verdict(track, frame, frame_id)

        # M01 规则降级为佐证：crop 不认为是 M01 时，规则候选不再单独成立
        if label == "M01" and classifier.m01_require_crop and pred_label != "M01":
            return base_label, base_confidence, "crop_gate_miss"

        new_label, new_conf, new_src, _action = decide_crop_action(
            label, confidence, src_tag, pred_label, pred_conf,
            review_classes=classifier.review_classes,
            thresholds=classifier.class_thresholds,
            normal_conf=classifier.normal_conf,
            confirm_alerts=classifier.confirm_alerts,
            base_label=base_label,
            base_confidence=base_confidence,
        )
        return new_label, new_conf, new_src

    def _geom_veto_allowed(self, track_id: str, cached_frame_id: int, strong_geom: bool) -> bool:
        """强几何 H03：两次不同缓存结论同向否决才允许压制。

        同一份缓存被连续读到多次只算一票，避免"一帧复核结果压掉整段真跌倒"。
        """
        if not strong_geom:
            return True
        last = self._geom_veto_last_cache.get(track_id, -1)
        if last != int(cached_frame_id):
            self._geom_veto_last_cache[track_id] = int(cached_frame_id)
            self._geom_veto_streak[track_id] = self._geom_veto_streak.get(track_id, 0) + 1
        return self._geom_veto_streak.get(track_id, 0) >= 2

    def _take_review_token(self) -> bool:
        """全局复核令牌桶：限制 VLM 复核频率，避免抢占检测推理的 GPU。"""
        if self._review_rate_per_min <= 0:
            return True
        now = time.monotonic()
        elapsed = max(0.0, now - self._review_token_ts)
        self._review_token_ts = now
        self._review_tokens = min(4.0, self._review_tokens + elapsed * self._review_rate_per_min / 60.0)
        if self._review_tokens >= 1.0:
            self._review_tokens -= 1.0
            return True
        return False

    def _submit_async_vlm_review(self, track: dict, frame: np.ndarray, frame_id: int,
                                  pc_label: str, pc_conf: float, context: str = "") -> None:
        """提交异步 VLM 复核：立即返回，VLM 完成后回写 action_cache + vlm_review_cache。"""
        track_id = str(track["track_id"])
        if not self._take_review_token():
            self._review_rate_limited += 1
            return
        # 准备 person crop 帧——问题1：优先原帧缓冲（真实细节翻倍），无原帧回退降采样缓冲
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        review_bbox = list(bbox)  # 闭包捕获：回调登记空间确认/否决用
        x, y, w, h = [int(round(v)) for v in bbox]
        frames = []
        src_buffer = self.frame_buffer
        sx = sy = 1.0
        if len(self.frame_buffer_orig) > 0:
            src_buffer = self.frame_buffer_orig
            proc_h, proc_w = (frame.shape[:2] if frame is not None else (540, 960))
            # 以最近一条原帧的实际尺寸换算（兼容不同分辨率源）
            _last = self.frame_buffer_orig.sample(1)
            if _last:
                orig_h, orig_w = _last[0][1].shape[:2]
                sx, sy = orig_w / max(1, proc_w), orig_h / max(1, proc_h)
        # 主动选帧：吸烟/手机类复核优先取该 track 手-嘴距离最近的历史帧（动作峰值更可能含烟）
        sampled = []
        if pc_label in ("M01", "M02"):
            try:
                sampled = self._select_review_frames_by_hand_mouth(track_id, bbox, k=3, buffer=src_buffer)
            except Exception:
                sampled = []
        if not sampled:
            sampled = src_buffer.sample(3)
        for _, full_f in sampled:
            h_f, w_f = full_f.shape[:2]
            _x, _y, _w, _h = x * sx, y * sy, w * sx, h * sy
            x1, y1 = max(0, int(_x)), max(0, int(_y))
            x2, y2 = min(w_f, int(_x + _w)), min(h_f, int(_y + _h))
            crop = full_f[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else full_f
            if crop.size > 0:
                frames.append(crop)
        if len(frames) < 2 and frame is not None:
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_f, x + w), min(h_f, y + h)
            crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame
            if crop.size > 0:
                frames = [crop]
        if not frames:
            return

        # H03 复核必须看到"人与地面的关系"：纯局部 crop 里躺卧与蹲坐、近景站立不可分，
        # 实测 laying 视频连续 3 次复核均返回 Normal conf=0.95。首帧补一张全景缩略图并画红框。
        if pc_label == "H03" and frame is not None:
            oh, ow = frame.shape[:2]
            _scale = min(1.0, 640.0 / max(1.0, float(ow)))
            pano = cv2.resize(frame, (max(1, int(ow * _scale)), max(1, int(oh * _scale))))
            px, py = pano.shape[1] / max(1.0, ow), pano.shape[0] / max(1.0, oh)
            cv2.rectangle(pano, (int(x * px), int(y * py)),
                          (int((x + w) * px), int((y + h) * py)), (0, 0, 255), 2)
            # 2 图（全景 + 1 张裁剪）足够判"人与地面关系"，比 3 图快约 1/3，
            # 复核吞吐是这条链路的瓶颈（见 _review_inflight 注释）
            frames = [pano] + frames[:1]

        # 小 crop 上采样：960px 降采样源帧裁出的 person crop 常 <100px，VLM 视觉编码
        # 在该尺度下丢失细粒度动作证据（吸烟/手机），放大到短边 ≥256px 再送审
        upscaled = []
        for f in frames:
            fh, fw = f.shape[:2]
            short_side = min(fh, fw)
            if short_side > 0 and short_side < 256:
                scale = 256.0 / short_side
                f = cv2.resize(f, (max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))),
                               interpolation=cv2.INTER_CUBIC)
            upscaled.append(f)
        frames = upscaled

        # B5: 缓存检查——相同 crop 直接复用上次 VLM 结果
        cached_result = self._vlm_cache.get(frames[0])
        if cached_result is not None:
            with self._cache_lock:
                self.action_cache[track_id] = cached_result
                self.track_vlm_review_cache[track_id] = (int(frame_id), cached_result[0], cached_result[1])
                self._vlm_cache_sourced.add(track_id)  # 归因：该 cache 值源自 VLM
            return

        def callback(result: dict) -> None:
            """VLM 完成后回写缓存，下一帧自动使用复核结果。"""
            if result.get("status") == "success" and result.get("label"):
                new_label = str(result["label"])
                vlm_label = new_label  # 原始裁决（平滑会把它改回 pc_label，确认登记必须用原值）
                new_conf = float(result.get("confidence", pc_conf))
                # 双钥匙 B：VLM 单方面 Normal→M01 升级（无规则候选）不采信——防办公场景幻觉种子；
                # 规则触发后的复核确认（pc_label==M01，双钥匙第二把）直接采信，误报由规则几何把关
                if new_label == "M01" and pc_label != "M01":
                    new_label, new_conf = pc_label, pc_conf
                # 判定平滑：多数投票（最近3次≥2票）才允许改变方向，防单次抖动翻转状态
                if new_label != pc_label:
                    hist = self._review_history[track_id]
                    hist.append(new_label)
                    if len(hist) >= 2 and sum(1 for x in hist if x == new_label) >= 2:
                        pass  # 连续2次同向 → 采纳改变
                    else:
                        new_label, new_conf = pc_label, pc_conf  # 首次翻转 → 本次不采信，维持原判
                else:
                    self._review_history[track_id].append(new_label)
                # 归因诊断：判定留痕
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    _vd = _Path("results/vlm_verdicts.jsonl")
                    with _vd.open("a", encoding="utf-8") as _vf:
                        _vf.write(_json.dumps({
                            "frame_id": int(frame_id), "track_id": track_id,
                            "claimed": pc_label, "returned": new_label, "conf": new_conf,
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                with self._cache_lock:
                    self.action_cache[track_id] = (new_label, new_conf)
                    self.track_vlm_review_cache[track_id] = (int(frame_id), new_label, new_conf)
                    self._vlm_cache_sourced.add(track_id)  # 归因：该 cache 值源自 VLM
                    # 归因修复：复核结论与声称不一致 → 否决锁存（连续2次同向才生效，防单次抖动误锁）
                    # 用 vlm_label：平滑阶段已把单次否决改回 pc_label，用 new_label 判断会永远进不来
                    if vlm_label != pc_label and pc_label in self.force_vlm_classes:
                        veto_frames = int(self._vlm_veto_seconds * max(1.0, self._source_fps))
                        veto_hist = self._veto_history.setdefault(track_id, deque(maxlen=2))
                        veto_hist.append(vlm_label)
                        if len(veto_hist) >= 2 and all(x == vlm_label for x in veto_hist):
                            self._vlm_veto.setdefault(track_id, {})[pc_label] = int(frame_id) + veto_frames
                            if review_bbox is not None:
                                self._vlm_veto_spatial.setdefault(self._spatial_key(review_bbox), {})[pc_label] = int(frame_id) + veto_frames
                        else:
                            new_label, new_conf = pc_label, pc_conf  # 单次否决不生效也不回写
                    # 双钥匙 A：复核明确确认某高危类 → 登记该类的确认窗口
                    # （原实现写死 M01，导致 H03 规则候选永远拿不到第二把钥匙、始终 rule_pending）
                    if vlm_label == pc_label and pc_label in self.force_vlm_classes:
                        hold = int(self._vlm_confirm_hold_seconds * max(1.0, self._source_fps))
                        until = int(frame_id) + hold
                        self._vlm_confirmed[self._confirm_key(track_id, pc_label)] = until
                        if review_bbox is not None:
                            # 空间确认登记 3x3 邻域：倒地到复核返回有数秒延迟，这期间人已小幅移动
                            # 且 tracker 常换 ID（实测 P004 确认后规则候选落到 P001），单格锚定会丢钥匙
                            gx, gy = self._spatial_key(review_bbox)
                            for _dx in (-1, 0, 1):
                                for _dy in (-1, 0, 1):
                                    key = ((gx + _dx, gy + _dy), pc_label)
                                    self._vlm_confirmed_spatial[key] = max(
                                        until, self._vlm_confirmed_spatial.get(key, 0))
                if frames:
                    self._vlm_cache.put(frames[0], new_label, new_conf)

        prompt = build_forced_prompt(pc_label)
        # 在飞名额：vLLM 吞吐远低于提交速率（单次 3 图复核 ~10s，引擎 12fps 每帧都可能提交），
        # 无上限时队列会无限堆积——实测 live 任务中所有复核都卡在第 60 帧之前的候选上，
        # 跌倒发生时的候选永远排不到执行，表现为"H03 规则命中但永不确认"。
        inflight_key = (track_id, str(pc_label))
        if inflight_key in self._review_inflight or len(self._review_inflight) >= self._review_inflight_max:
            return
        self._review_inflight.add(inflight_key)
        self._vlm_async_executor.submit(self._async_vlm_review_job, frames, track_id, context,
                                        callback, prompt, inflight_key)

    def _async_vlm_review_job(self, frames: list, track_id: str, context: str,
                              callback, prompt: str | None = None,
                              inflight_key: tuple[str, str] | None = None) -> None:
        """异步 VLM 复核工作线程。"""
        # B3: 全局并发槽位——复核类调用阻塞排队（10s 超时后放弃，PC 结果兜底）
        acquired = self.vlm_router.try_acquire(timeout=10.0)
        try:
            if not acquired:
                return
            result = self.vlm_reviewer.review_clip(
                frames=frames, track_id=track_id, prompt=prompt, context=context,
            )
            callback(result)
        except Exception:
            # VLM 失败 = 不修正（PC 结果留在 cache 里，已经是对的）
            pass
        finally:
            if inflight_key is not None:
                self._review_inflight.discard(inflight_key)
            if acquired:
                self.vlm_router.release()

    def _forced_vlm_review(self, track: dict, frame, frame_id: int, label: str, conf: float) -> tuple[str, float]:
        """对强制复核类别（如 H03）做 VLM 确认；同目标节流，失败回退原结果。"""
        track_id = str(track["track_id"])
        cached = self.track_vlm_review_cache.get(track_id)
        if cached is not None and (frame_id - cached[0]) < self.force_vlm_throttle_frames:
            return cached[1], cached[2]
        # person crop 模式：从环形缓冲全帧按 bbox 裁 3 帧高清 crop
        # （吸烟/跌倒等细粒度行为需要看清手部与烟，全场景 480px 不够）
        bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = [int(round(v)) for v in bbox]
        frames = []
        sampled = self.frame_buffer.sample(3)
        for _, full_f in sampled:
            h_f, w_f = full_f.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_f, x + w), min(h_f, y + h)
            crop = full_f[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else full_f
            if crop.size > 0:
                frames.append(crop)
        if len(frames) < 2:
            if frame is not None:
                h_f, w_f = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_f, x + w), min(h_f, y + h)
                crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame
                if crop.size > 0:
                    frames = [crop]
        # 全景缩略（红框标目标）+ 局部 crop：只看局部时 VLM 无法判断人与地面的关系，
        # 真实跌倒会被误判 Normal（实测 laying 视频 3 次复核均返回 Normal conf=0.95）。
        if sampled:
            _, full_ref = sampled[-1]
            oh, ow = full_ref.shape[:2]
            scale = min(1.0, 640.0 / max(1.0, float(ow)))
            small = cv2.resize(full_ref, (max(1, int(ow * scale)), max(1, int(oh * scale))))
            sx, sy = small.shape[1] / max(1.0, ow), small.shape[0] / max(1.0, oh)
            cv2.rectangle(small, (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)), (0, 0, 255), 2)
            frames = [small] + frames[:2]

        result = self.vlm_reviewer.review_clip(
            frames=frames,
            track_id=track_id,
            context=self._build_review_context(track_id),
        )
        if result.get("status") == "success" and result.get("label"):
            label = str(result["label"])
            conf = float(result.get("confidence", conf))
        self.track_vlm_review_cache[track_id] = (int(frame_id), label, conf)
        return label, conf

    def _build_review_context(self, track_id: str) -> str:
        """组装灰区复核上下文：场景叙述摘要 + 该目标近期行为序列。"""
        if not getattr(self, "review_context_enabled", True):
            return ""
        parts: list[str] = []
        scene_ctx = self.scene_narrator.get_context()
        if scene_ctx:
            parts.append(scene_ctx)
        history = self.track_label_history.get(track_id)
        if history:
            seq = " → ".join(f"{label}({conf:.2f})" for label, conf in list(history)[-12:])
            parts.append(f"该目标近期模型输出序列：{seq}")
        return "\n".join(parts)

    def _motion_score(self, frame) -> float:
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._last_detector_gray is None:
            self._last_detector_gray = gray
            return 1.0

        diff = cv2.absdiff(gray, self._last_detector_gray)
        self._last_detector_gray = gray
        return float(np.mean(diff) / 255.0)

    def _should_run_detector(self, frame_id: int, motion_score: float, has_tracks: bool) -> bool:
        if not self.enable_detector_gating:
            return True
        if frame_id <= self.detector_warmup_frames:
            return True
        if not has_tracks:
            return True
        if self._detector_skip_count >= self.detector_max_skip_frames:
            return True
        if motion_score >= self.detector_motion_threshold:
            return True
        if motion_score <= self.detector_low_motion_threshold and (frame_id % self.detector_stride) != 0:
            return False
        return (frame_id % self.detector_stride) == 0

    def _should_refresh_pose(self, frame_id: int, track: dict, track_age: int) -> bool:
        track_id = track["track_id"]
        if not self.enable_pose_motion_gate:
            return False

        if track_id not in self.pose_cache:
            return True

        last_pose_frame = int(self.pose_last_frame.get(track_id, 0))
        if last_pose_frame > 0 and (frame_id - last_pose_frame) >= self.pose_force_refresh_gap:
            return True

        if (frame_id % self.pose_stride) == 0:
            return True

        prev_bbox = self.pose_last_bbox.get(track_id)
        if prev_bbox is None:
            return False

        iou = self._bbox_iou_xywh(prev_bbox, track.get("bbox", [0, 0, 0, 0]))
        if iou < self.pose_motion_gate_iou_threshold:
            return True
        if track_age <= self.action_min_track_age + 1:
            return True
        return False

    def process(self, frame, frame_id: int, orig_frame=None) -> FrameResult:
        """同步推理一帧：render / 评估 / 管道禁用时使用（行为与重构前一致）。"""
        ctx = self._make_ctx(frame, frame_id)
        if orig_frame is not None:
            ctx["orig_frame"] = orig_frame
        self._stage1_detect(ctx)
        self._stage2_pose(ctx)
        self._stage3_recognize(ctx)
        return self._assemble_result(ctx)

    # ── B6: 三阶段流水线 ─────────────────────────────────────
    # 同步路径（render/评估/管道禁用）：process() 顺序调用四个阶段方法，行为与旧版完全一致。
    # 管道路径：三个工作线程各自消费队列跑一个阶段，帧间重叠执行
    # （stage1 跑第 N+1 帧检测时，stage2 跑第 N 帧姿态、stage3 跑第 N-1 帧行为）。
    # 跨阶段状态通过 ctx dict 传递；各阶段只独占自己的模块，共享 dict 均为单键原子操作。

    def _make_ctx(self, frame, frame_id: int) -> dict:
        return {"frame": frame, "frame_id": frame_id, "start": time.perf_counter()}

    def _stage1_detect(self, ctx: dict) -> None:
        """阶段1：运动门控 + 检测/跟踪 + C1 ID切换检测。"""
        frame = ctx["frame"]
        frame_id = ctx["frame_id"]

        stage_detector_t0 = time.perf_counter()
        motion_score = self._motion_score(frame)
        run_detector = self._should_run_detector(frame_id, motion_score, bool(self.track_seen_frames))
        if run_detector:
            detections_raw = self.detector.infer(frame)
            tracks = self.tracker.update(detections_raw)
            self._detector_skip_count = 0
        else:
            tracks = self.tracker.advance_without_detections()
            self._detector_skip_count += 1
        stage_detector_ms = (time.perf_counter() - stage_detector_t0) * 1000.0

        # C1: ID 切换检测——bbox 跳变时清零该 track 的逐目标缓存（防混人序列污染）
        self._detect_and_handle_id_switches(tracks)
        ctx["tracks"] = tracks
        ctx["run_detector"] = run_detector
        ctx["stage_detector_ms"] = stage_detector_ms

    def _stage2_pose(self, ctx: dict) -> None:
        """阶段2：track 状态装配 + 姿态批次推理（pose_cache 更新）。"""
        frame = ctx["frame"]
        frame_id = ctx["frame_id"]
        tracks = ctx["tracks"]
        stage_pose_ms = 0.0
        stage_action_ms = 0.0

        detections: list[Detection] = []
        # B1: VLM 异步复核——本帧标记 pending，VLM 完成后下一帧从 cache 自动更新
        vlm_review = VLMReview(enabled=self.vlm_reviewer.enabled, status="pending" if self.vlm_reviewer.enabled else "skipped")

        main_label = "Normal"
        main_confidence = 0.0
        pending_tracks: list[dict] = []
        sorted_tracks = sorted(tracks, key=lambda item: float(item.get("score", 0.0)), reverse=True)

        pose_compute_set = set()
        if (frame_id % self.pose_stride) == 0:
            pose_compute_set = {item.get("track_id") for item in sorted_tracks[: self.max_pose_tracks]}

        action_candidates = []
        for item in sorted_tracks:
            score = float(item.get("score", 0.0))
            if score < self.action_min_track_score:
                continue
            action_candidates.append(item)
            if len(action_candidates) >= self.max_action_tracks:
                break
        action_compute_set = {item.get("track_id") for item in action_candidates}

        action_runs = 0
        pose_skip_motion_gate = 0
        action_skip_warmup = 0
        action_skip_track_age = 0
        action_skip_gate = 0
        action_skip_cached = 0
        action_skip_default = 0
        rule_override_count = 0
        reasoner_source_counts = {"action_model": 0, "rule_engine": 0}
        reasoner_rule_tag_counts: dict[str, int] = {}
        h01_h04_mapping = {
            "base_h01": 0,
            "base_h04": 0,
            "refined_h01": 0,
            "refined_h04": 0,
            "base_h01_conf_sum": 0.0,
            "base_h04_conf_sum": 0.0,
            "refined_h01_conf_sum": 0.0,
            "refined_h04_conf_sum": 0.0,
        }

        track_states: list[dict[str, object]] = []
        pose_batch_tracks: list[dict] = []
        pose_batch_state_indices: list[int] = []

        for track in tracks:
            track_id = track["track_id"]
            self.track_seen_frames[track_id] = self.track_seen_frames.get(track_id, 0) + 1
            track_age = self.track_seen_frames[track_id]

            if self.enable_pose_motion_gate:
                should_pose_infer = self._should_refresh_pose(frame_id, track, track_age)
                if (
                    not should_pose_infer
                    and track_id in self.pose_cache
                    and track_id in pose_compute_set
                    and self.pose_motion_gate_iou_threshold <= 1.0
                ):
                    prev_bbox = self.pose_last_bbox.get(track_id)
                    if prev_bbox is not None:
                        iou = self._bbox_iou_xywh(prev_bbox, track.get("bbox", [0, 0, 0, 0]))
                        if iou >= self.pose_motion_gate_iou_threshold:
                            should_pose_infer = False
                            pose_skip_motion_gate += 1
            else:
                should_pose_infer = track_id in pose_compute_set or track_id not in self.pose_cache

            state = {
                "track": track,
                "track_id": track_id,
                "track_age": track_age,
                "keypoints": self.pose_cache.get(track_id, []),
                "pose_should_infer": should_pose_infer,
                "label": self.action_cache.get(track_id, (self.action_default_label, self.action_default_confidence))[0],
                "confidence": self.action_cache.get(track_id, (self.action_default_label, self.action_default_confidence))[1],
            }
            track_states.append(state)

            if should_pose_infer:
                pose_batch_state_indices.append(len(track_states) - 1)
                pose_batch_tracks.append(track)

        if pose_batch_tracks:
            pose_t0 = time.perf_counter()
            pose_batch_results = self.pose_estimator.infer_batch(frame, pose_batch_tracks)
            stage_pose_ms += (time.perf_counter() - pose_t0) * 1000.0
            for state_index, keypoints in zip(pose_batch_state_indices, pose_batch_results):
                state = track_states[state_index]
                track = state["track"]
                track_id = str(state["track_id"])
                keypoints = keypoints or []
                self.pose_cache[track_id] = keypoints
                self.pose_last_bbox[track_id] = list(track.get("bbox", [0, 0, 0, 0]))
                self.pose_last_frame[track_id] = frame_id
                state["keypoints"] = keypoints
        ctx["detections"] = detections
        ctx["vlm_review"] = vlm_review
        ctx["main_label"] = main_label
        ctx["main_confidence"] = main_confidence
        ctx["pending_tracks"] = pending_tracks
        ctx["track_states"] = track_states
        ctx["stage_pose_ms"] = stage_pose_ms
        ctx["stage_action_ms"] = stage_action_ms
        ctx["action_compute_set"] = action_compute_set
        ctx["action_runs"] = action_runs
        ctx["pose_skip_motion_gate"] = pose_skip_motion_gate
        ctx["rule_override_count"] = rule_override_count
        ctx["reasoner_source_counts"] = reasoner_source_counts
        ctx["reasoner_rule_tag_counts"] = reasoner_rule_tag_counts
        ctx["h01_h04_mapping"] = h01_h04_mapping
        ctx["action_skip_warmup"] = action_skip_warmup
        ctx["action_skip_track_age"] = action_skip_track_age
        ctx["action_skip_gate"] = action_skip_gate
        ctx["action_skip_cached"] = action_skip_cached
        ctx["action_skip_default"] = action_skip_default

    def _stage3_recognize(self, ctx: dict) -> None:
        """阶段3：三级决策 + 规则精化 + C2平滑 + 告警/场景旁路 + 姿态缓冲。"""
        frame = ctx["frame"]
        frame_id = ctx["frame_id"]
        tracks = ctx["tracks"]
        start = ctx["start"]
        stage_action_ms = ctx["stage_action_ms"]
        track_states = ctx["track_states"]
        action_compute_set = ctx["action_compute_set"]
        detections = ctx["detections"]
        vlm_review = ctx["vlm_review"]
        main_label = ctx["main_label"]
        main_confidence = ctx["main_confidence"]
        pending_tracks = ctx["pending_tracks"]
        action_runs = ctx["action_runs"]
        pose_skip_motion_gate = ctx["pose_skip_motion_gate"]
        rule_override_count = ctx["rule_override_count"]
        reasoner_source_counts = ctx["reasoner_source_counts"]
        reasoner_rule_tag_counts = ctx["reasoner_rule_tag_counts"]
        h01_h04_mapping = ctx["h01_h04_mapping"]
        action_skip_warmup = ctx["action_skip_warmup"]
        action_skip_track_age = ctx["action_skip_track_age"]
        action_skip_gate = ctx["action_skip_gate"]
        action_skip_cached = ctx["action_skip_cached"]
        action_skip_default = ctx["action_skip_default"]
        action_batch_tracks: list[dict] = []
        action_batch_keypoints: list[list[list[float]]] = []
        action_batch_state_indices: list[int] = []

        for state_index, state in enumerate(track_states):
            track = state["track"]
            track_id = str(state["track_id"])
            track_age = int(state["track_age"])
            keypoints = state["keypoints"]

            can_run_action = True
            if frame_id <= self.action_warmup_frames:
                can_run_action = False
                action_skip_warmup += 1
            elif track_age < self.action_min_track_age:
                can_run_action = False
                action_skip_track_age += 1
            elif track_id not in action_compute_set:
                can_run_action = False
                action_skip_gate += 1

            should_run_by_stride = (frame_id % self.action_stride) == 0 or track_id not in self.action_cache
            if can_run_action and should_run_by_stride:
                action_batch_state_indices.append(state_index)
                action_batch_tracks.append(track)
                action_batch_keypoints.append(keypoints)
            elif track_id in self.action_cache:
                state["label"], state["confidence"] = self.action_cache[track_id]
                action_skip_cached += 1
            else:
                state["label"] = self.action_default_label
                state["confidence"] = self.action_default_confidence
                self.action_cache[track_id] = (self.action_default_label, self.action_default_confidence)
                action_skip_default += 1

        # D4: 时间粒度对齐——每个处理帧都为所有 track 追加姿态到缓冲
        # （与训练数据的连续帧口径一致；此前只在 action 步长帧追加，时间粒度不匹配）
        frame_h, frame_w = frame.shape[:2] if frame is not None else (1080, 1920)
        # B6: keypoints 随 ctx 传递，stage3 不跨线程读 stage2 的 pose_cache
        for _state in track_states:
            self.action_recognizer.append_pose(str(_state["track_id"]), _state["keypoints"],
                                               frame_width=frame_w, frame_height=frame_h)

        batch_label_sources: dict[str, str] = {}
        if action_batch_tracks:
            action_t0 = time.perf_counter()
            # ── 三级决策：PoseClassifier 主路径 ──────────────────
            if self.action_recognizer.pose_classifier_loaded:
                # Step 1: PoseClassifier 批次推理
                try:
                    pc_results = self.action_recognizer.infer_pose_classifier_batch(
                        [str(s["track_id"]) for s in action_batch_tracks]
                    )
                except Exception:
                    # 姿态缓冲形状异常（如 17/133 混入）时兜底，不崩溃
                    pc_results = [("Normal", 0.0) for _ in action_batch_tracks]
                # Step 2: 三级决策
                three_level_results: list[tuple[str, float]] = []
                vlm_fallback_tracks: list[dict] = []
                vlm_fallback_indices: list[int] = []
                batch_label_sources: dict[str, str] = {}
                for state_idx, (track, pc_result) in enumerate(zip(action_batch_tracks, pc_results)):
                    pc_label, pc_conf = pc_result
                    src_tag = "pc_high" if pc_conf >= self.action_recognizer.pc_conf_high else (
                        "pc_gray" if pc_conf >= self.action_recognizer.pc_conf_low else "slowfast")
                    # B1: 强制复核改为异步——立即用 PC 结果，复核回写 cache 供下一帧
                    # 例外：H03 不走 PC 侧复核。PC 给的 H03 置信度普遍 0.4~0.63（实测），
                    # 每帧提交只会挤占 vLLM 队列，把真正几何命中的复核压在后面；
                    # H03 的复核统一由规则门（几何命中）发起。
                    if pc_label in self.force_vlm_classes and pc_label != "H03":
                        cached = self.track_vlm_review_cache.get(str(track["track_id"]))
                        if cached is not None and (frame_id - cached[0]) < self.force_vlm_throttle_frames:
                            pc_label, pc_conf = cached[1], cached[2]
                            src_tag = "vlm_cache"  # 归因：取自复核节流缓存
                        else:
                            self._submit_async_vlm_review(track, frame, frame_id, pc_label, pc_conf,
                                                          context=self._build_review_context(str(track["track_id"])))
                    batch_label_sources[str(track["track_id"])] = src_tag
                    if pc_conf >= self.action_recognizer.pc_conf_high:
                        # 级别1：高置信度，直接采纳
                        three_level_results.append((pc_label, pc_conf))
                    elif pc_conf >= self.action_recognizer.pc_conf_low:
                        # 级别2：灰色地带 → 异步 VLM 复核（回写 cache 供下一帧）
                        three_level_results.append((pc_label, pc_conf))
                        # 只复核"有具体标签"的灰区帧：PC=Normal 的灰区帧送审价值极低
                        # （回写时 Normal→M01 本来就不采信），却会持续占用共享 GPU，
                        # 实测 60s 内 34 次复核把检测吞吐从 ~724 帧压到 ~505 帧。
                        if pc_label != "Normal":
                            vlm_fallback_tracks.append(track)
                            vlm_fallback_indices.append(state_idx)
                    else:
                        # 级别3：低置信度 → SlowFast 关键词映射兜底
                        sf_result = self.action_recognizer.infer(
                            track, frame_id,
                            action_batch_keypoints[state_idx] if state_idx < len(action_batch_keypoints) else None,
                            frame,
                        )
                        three_level_results.append(sf_result)

                # B1: 灰区 VLM 复核改为异步——立即用 PC 结果，VLM 完成后回写 cache
                if vlm_fallback_tracks and self.vlm_reviewer.enabled:
                    for vlm_idx, track in enumerate(vlm_fallback_tracks):
                        track_id = str(track["track_id"])
                        # 提取 person crop 帧用于 VLM 时序上下文
                        vlm_frames = []
                        if frame is not None:
                            bbox = [float(v) for v in track.get("bbox", [0, 0, 0, 0])]
                            x, y, w, h = [int(round(v)) for v in bbox]
                            h_f, w_f = frame.shape[:2]
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(w_f, x + w), min(h_f, y + h)
                            crop = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame
                            if crop.size > 0:
                                vlm_frames.append(crop)

                        # 异步提交：PC 结果保留在本帧，VLM 回写 action_cache 供下一帧
                        self._submit_async_vlm_review(track, frame, frame_id,
                                                      three_level_results[vlm_fallback_indices[vlm_idx]][0],
                                                      three_level_results[vlm_fallback_indices[vlm_idx]][1],
                                                      context=self._build_review_context(track_id))

                action_batch_results = three_level_results
            else:
                # PoseClassifier 未加载：走 SlowFast 批量兜底
                action_batch_results = self.action_recognizer.infer_batch(
                    action_batch_tracks,
                    frame_id,
                    pose_keypoints_list=action_batch_keypoints,
                    frame=frame,
                )
            stage_action_ms += (time.perf_counter() - action_t0) * 1000.0
            action_runs += len(action_batch_results)
            for state_index, (label, confidence) in zip(action_batch_state_indices, action_batch_results):
                state = track_states[state_index]
                track_id = str(state["track_id"])
                state["label"] = label
                state["confidence"] = confidence
                self.action_cache[track_id] = (label, confidence)
                # 行为序列历史（供 VLM 复核上下文使用）
                history = self.track_label_history.setdefault(track_id, deque(maxlen=16))
                history.append((label, float(confidence)))

        label_sources: dict[str, str] = {}
        for state in track_states:
            track = state["track"]
            track_id = str(state["track_id"])
            keypoints = state["keypoints"]
            label = str(state["label"])
            confidence = float(state["confidence"])
            base_label = label
            base_confidence = confidence
            # 归因：初始来源——批次内标记；非批次 track 为缓存沿用（区分是否源自 VLM 回写）
            src_tag = batch_label_sources.get(track_id)
            if src_tag is None:
                src_tag = "vlm_async" if track_id in self._vlm_cache_sourced else "cache"

            if base_label == "H01":
                h01_h04_mapping["base_h01"] += 1
                h01_h04_mapping["base_h01_conf_sum"] += base_confidence
            elif base_label == "H04":
                h01_h04_mapping["base_h04"] += 1
                h01_h04_mapping["base_h04_conf_sum"] += base_confidence

            # 使用轻量规则推理（PoseClassifier 加载后仅位置规则）
            if self.action_recognizer.pose_classifier_loaded:
                refined_label, refined_confidence, source, rule_tags = self.behavior_reasoner.refine_light(
                    track_id=track_id,
                    bbox=track.get("bbox", [0, 0, 0, 0]),
                    keypoints=keypoints,
                    frame_shape=frame.shape[:2],
                    base_label=label,
                    base_confidence=confidence,
                    peer_tracks=tracks,
                    frame_id=frame_id,
                )
            else:
                refined_label, refined_confidence, source, rule_tags = self.behavior_reasoner.infer(
                    track_id=track_id,
                    bbox=track.get("bbox", [0, 0, 0, 0]),
                    keypoints=keypoints,
                    frame_shape=frame.shape[:2],
                    base_label=label,
                    base_confidence=confidence,
                    peer_tracks=tracks,
                    frame_id=frame_id,
                )
            label, confidence = refined_label, refined_confidence
            if source == "rule_engine":
                rule_override_count += 1
                src_tag = "rule"  # 归因：规则引擎覆盖

            # ── 本地 crop 分类器：M01 主提案 + 规则佐证门控 ──
            # 1) 提案：Normal 帧上连续 N 次判 M01 → 生成 M01 候选（走规则复核门，VLM 确认后才告警）
            # 2) 门控/否决：pose 规则的 M01 候选若 crop 不认可 → 撤销；明显 Normal 的候选直接按掉
            label, confidence, src_tag, _crop_proposed = self._maybe_crop_proposal(
                track, frame, frame_id, label, confidence, src_tag, base_label, base_confidence)
            if _crop_proposed:
                source = "rule_engine"  # 提案按规则候选处理：继续走 VLM 确认门
            label, confidence, src_tag = self._apply_crop_review(
                track, frame, frame_id, label, confidence, src_tag, base_label, base_confidence)

            # 显示层兜底：未确认的规则候选（rule_pending）不进入告警，但应以候选标签展示，
            # 避免"人已倒地、画面还是绿框 Normal"。仅影响可视化，不影响告警口径。
            display_label, display_conf, display_pending = "", 0.0, False

            # 归因修复 + 双钥匙：规则触发的高危类必须经 VLM 复核；M01 需"规则候选 ∧ VLM确认"双钥匙
            if label in self.force_vlm_classes and source == "rule_engine":
                rule_candidate_conf = float(confidence)
                rule_label = label  # 规则候选标签：后面 label 会被重置为 base_label，确认窗口必须用原值
                # 强几何 H03（P1-5）：倒地只有几秒，不能等 VLM 再告警；
                # 但并行送审，迟到的否决在后续帧生效（veto 锁存），兼顾时延与纠错。
                strong_geom = (
                    rule_label == "H03"
                    and "fall_geometry_rule" in rule_tags
                    and self.behavior_reasoner.is_strong_fall_geometry(
                        track.get("bbox", [0, 0, 0, 0]), frame.shape[:2], keypoints)
                )
                geom_strong = strong_geom and self.geom_strong_parallel_review
                skey = self._spatial_key(track.get("bbox", [0, 0, 0, 0]))
                veto_map = {**self._vlm_veto_spatial.get(skey, {}),
                             **self._vlm_veto.get(track_id, {})}
                cached = self.track_vlm_review_cache.get(track_id)
                if frame_id < int(veto_map.get(label, 0)):
                    # 锁存期内：VLM 近期已否决该类 → 压制规则输出
                    label, confidence = base_label, base_confidence
                    src_tag = "vlm_veto"
                elif cached is not None and (frame_id - cached[0]) < self._veto_cache_frames:
                    if cached[1] != label:
                        # 复核缓存近期否决 → 登记锁存并压制。
                        # 强几何 H03 例外：需要两次"不同缓存结论"同向否决，避免单次误判压掉真跌倒。
                        if self._geom_veto_allowed(track_id, int(cached[0]), strong_geom):
                            veto_frames = int(self._vlm_veto_seconds * max(1.0, self._source_fps))
                            self._vlm_veto.setdefault(track_id, {})[label] = frame_id + veto_frames
                            label, confidence = base_label, base_confidence
                            src_tag = "vlm_veto"
                    else:
                        # 复核确认 → 清空强几何否决计数
                        self._geom_veto_last_cache.pop(track_id, None)
                        self._geom_veto_streak.pop(track_id, None)
                else:
                    # 提交规则触发的复核（claimed = 规则标签，复用 person-crop 通道）
                    if strong_geom:
                        self._geom_strong_reviews += 1
                    self._submit_async_vlm_review(track, frame, frame_id, label, confidence,
                                                  context=self._build_review_context(track_id))
                # 双钥匙 A：确认窗口按"当前类别"取 track 与空间键的 max（任一有效即放行，抗 ID 切换与判定抖动）
                hist = list(self._review_history.get(track_id, []))
                conf_key = self._confirm_key(track_id, rule_label)
                spatial_conf_key = (skey, rule_label)
                confirmed_until = max(int(self._vlm_confirmed.get(conf_key, 0)),
                                      int(self._vlm_confirmed_spatial.get(spatial_conf_key, 0)))
                # geom_strong 已在上面计算：强几何 H03 不等待 VLM（倒地过程只有几秒），
                # 但仍会走下面的异步送审；迟到的否决用 veto 锁存纠正后续帧。
                # crop_confirmed：本地 2D 分类器已确认（confirm_alerts 开启时）→ 不等 VLM
                _crop_key_ok = str(src_tag).startswith("crop_confirmed")
                if frame_id > confirmed_until and not geom_strong and not _crop_key_ok:
                    recent = hist[-3:]
                    if sum(1 for x in recent if x == rule_label) >= 2:
                        hold = int(self._vlm_confirm_hold_seconds * max(1.0, self._source_fps))
                        self._vlm_confirmed[conf_key] = frame_id + hold  # 自续期
                        self._vlm_confirmed_spatial[spatial_conf_key] = max(
                            frame_id + hold, int(self._vlm_confirmed_spatial.get(spatial_conf_key, 0)))
                    else:
                        label, confidence = base_label, base_confidence
                        src_tag = "rule_pending"
                        if str(rule_label).startswith("H"):
                            display_label = str(rule_label)
                            display_conf = float(rule_candidate_conf)
                            display_pending = True
            reasoner_source_counts[source] = int(reasoner_source_counts.get(source, 0)) + 1
            for tag in rule_tags:
                reasoner_rule_tag_counts[tag] = int(reasoner_rule_tag_counts.get(tag, 0)) + 1

            # ── C2: 时序标签平滑（多数投票，基于 refine 后的最终标签） ──
            smooth_hist = self.track_smooth_history.get(track_id)
            if smooth_hist is None:
                smooth_hist = deque(maxlen=self._smooth_window)
                self.track_smooth_history[track_id] = smooth_hist
            smooth_hist.append(label)
            if len(smooth_hist) >= 3:
                vote_counts = Counter(smooth_hist)
                top_label, top_count = vote_counts.most_common(1)[0]
                if top_count >= self._smooth_min_votes and label != top_label:
                    label = top_label
                    confidence = max(confidence, 0.6)
                    src_tag += "+smooth"  # 归因：C2 平滑改判
            label_sources[track_id] = src_tag

            if label == "H01":
                h01_h04_mapping["refined_h01"] += 1
                h01_h04_mapping["refined_h01_conf_sum"] += confidence
            elif label == "H04":
                h01_h04_mapping["refined_h04"] += 1
                h01_h04_mapping["refined_h04_conf_sum"] += confidence
            risk_level = RISK_MAP.get(label, "LOW")

            # 主标签选择：先按风险等级，再按置信度。
            # 原实现纯比置信度，画面里一个 0.89 的 Normal 会压掉 0.78~0.9 的 H03，
            # 导致 build_alert 收到的 main_label 永远是 Normal——跌倒检测到了却不告警。
            # 风险加权只给"已通过复核门"的高危标签：rule_pending / vlm_veto 的标签
            # 只是未确认的规则候选（实测 conf 多为 0.55 的 PC 底数），加权会制造误报。
            _conf_floor = self.alert_system.high_risk_min_confidence
            # 只认"规则确认"或"VLM 确认"的来源。PC 的 H03（pc_gray/pc_high，实测
            # 置信度 0.55~0.95 都有）在就坐/弯腰时也会冒出来，加权会直接变成误报。
            # 注意 vlm_async 不可信：单次否决被平滑"改回"pc_label 后同样写进 cache 并标 vlm_async，
            # 于是 PC 的低置信 H03 会带着这个来源一路冒充"已确认"。
            _suppressed_src = not (src_tag.startswith("rule") and not src_tag.startswith("rule_pending"))
            _cur_prio = 2 if (RISK_MAP.get(main_label) == "HIGH" and main_confidence >= _conf_floor) else 1
            _new_prio = 2 if (risk_level == "HIGH" and confidence >= _conf_floor and not _suppressed_src) else 1
            if _new_prio > _cur_prio or (_new_prio == _cur_prio and confidence > main_confidence):
                main_label, main_confidence = label, confidence
            if label != "Normal" and risk_level in ("HIGH", "MEDIUM"):
                self._recent_abnormal[track_id] = {
                    "track_id": track_id, "label": label, "confidence": confidence,
                    "bbox": list(track.get("bbox", [0, 0, 0, 0])), "frame_id": frame_id,
                }
            pending_tracks.append(
                {
                    "track_id": track_id,
                    "bbox": list(track["bbox"]),
                    "keypoints": keypoints,
                    "label": label,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "label_source": src_tag,
                    "display_label": display_label or label,
                    "display_confidence": display_conf if display_pending else confidence,
                    "display_risk": RISK_MAP.get(display_label, risk_level) if display_pending else risk_level,
                    "display_pending": display_pending,
                    "activity_score": round(max(0.0, confidence - 0.45), 3),
                }
            )
            # ── VLM 发现器：事件触发 + 周期采样（异步，不阻塞主循环） ──
            try:
                self._maybe_submit_discovery(track, frame, frame_id, label, confidence, src_tag,
                                              tracks=tracks)
            except Exception:
                pass

        ctx["label_sources"] = label_sources
        role_map = self.interaction_analyzer.infer_roles(pending_tracks)

        temporal_result = self.temporal_adjudicator.update(
            frame_id=frame_id,
            pending_tracks=pending_tracks,
            frame_shape=frame.shape[:2],
            current_label=main_label,
            current_confidence=main_confidence,
        )
        main_label = str(temporal_result.get("output_label", main_label))
        main_confidence = float(temporal_result.get("output_confidence", main_confidence))

        for item in pending_tracks:
            role = role_map.get(item["track_id"], "unknown")
            detections.append(
                Detection(
                    track_id=item["track_id"],
                    bbox=item["bbox"],
                    pose_keypoints=item["keypoints"],
                    behavior=Behavior(
                        type=item.get("display_label") or item["label"],
                        confidence=item.get("display_confidence", item["confidence"]),
                        risk_level=item.get("display_risk", item["risk_level"]),
                        role=role,
                    ),
                )
            )

        stage_post_t0 = time.perf_counter()

        # ── 场景理解旁路（非阻塞）：环形缓冲入队 + 周期叙述 + 告警裁决进度 + 规则引擎 ──
        self.frame_buffer.push(frame, frame_id)
        _orig = ctx.get("orig_frame")
        if _orig is not None:
            self.frame_buffer_orig.push(_orig, frame_id)  # 问题1：原帧复核通道
        # A: 检测事实注入——当前快照 + 近窗异常聚合（叙述描述的是时间窗，参考也按窗口给，抗瞬时波动）
        window_seconds = float(getattr(self.scene_narrator, "interval_seconds", 20))
        window_frames = int(window_seconds * self._source_fps) if getattr(self, "_source_fps", None) else int(window_seconds * 25)
        injection = []
        for item in pending_tracks:
            entry = {"track_id": item["track_id"], "label": item["label"],
                     "confidence": item["confidence"], "bbox": item["bbox"]}
            recent = self._recent_abnormal.get(item["track_id"])
            if recent is not None and frame_id - int(recent.get("frame_id", 0)) <= window_frames:
                entry["label"] = recent["label"]
                entry["confidence"] = recent["confidence"]
            injection.append(entry)
        # ID 切换场景：近窗异常 track 若已不在当前帧（被重新编号），其事实仍须注入
        covered = {e["track_id"] for e in injection}
        for tid, recent in self._recent_abnormal.items():
            if tid in covered:
                continue
            if frame_id - int(recent.get("frame_id", 0)) <= window_frames:
                injection.append({
                    "track_id": tid, "label": recent["label"],
                    "confidence": recent["confidence"], "bbox": recent.get("bbox", [0, 0, 0, 0]),
                })
        self.scene_narrator.on_frame(frame_id, person_count=len(tracks),
                                     detections=injection, frame_shape=frame.shape[:2])
        self.alert_adjudicator.on_frame(frame_id)
        self.nl_rule_engine.on_frame(frame_id)
        self.novel_behavior_memo.on_frame(frame_id, pending_tracks)

        current_track_ids = {track["track_id"] for track in tracks}
        stale_ids = [track_id for track_id in self.pose_cache if track_id not in current_track_ids]
        for track_id in stale_ids:
            self.pose_cache.pop(track_id, None)
            self.pose_last_bbox.pop(track_id, None)
            self._crop_review_cache.pop(track_id, None)
            self._crop_proposal_hits.pop(track_id, None)
            self._geom_veto_last_cache.pop(track_id, None)
            self._geom_veto_streak.pop(track_id, None)
            self.pose_last_frame.pop(track_id, None)
        stale_action_ids = [track_id for track_id in self.action_cache if track_id not in current_track_ids]
        for track_id in stale_action_ids:
            self.action_cache.pop(track_id, None)
        stale_seen_ids = [track_id for track_id in self.track_seen_frames if track_id not in current_track_ids]
        for track_id in stale_seen_ids:
            self.track_seen_frames.pop(track_id, None)
        stale_prev_bbox_ids = [track_id for track_id in self._track_prev_bbox if track_id not in current_track_ids]
        for track_id in stale_prev_bbox_ids:
            self._track_prev_bbox.pop(track_id, None)
        stale_smooth_ids = [track_id for track_id in self.track_smooth_history if track_id not in current_track_ids]
        for track_id in stale_smooth_ids:
            self.track_smooth_history.pop(track_id, None)
        stale_history_ids = [track_id for track_id in self.track_label_history if track_id not in current_track_ids]
        for track_id in stale_history_ids:
            self.track_label_history.pop(track_id, None)
        stale_vlm_ids = [track_id for track_id in self.track_vlm_review_cache if track_id not in current_track_ids]
        for track_id in stale_vlm_ids:
            self.track_vlm_review_cache.pop(track_id, None)
            self._vlm_cache_sourced.discard(track_id)
            self._vlm_veto.pop(track_id, None)
            # 确认窗口/复核历史豁免：ID 切换时人的确认状态应延续（空间键兜底），只清易污染的序列缓存
            self._review_history.pop(track_id, None)
            self._veto_history.pop(track_id, None)
        self.action_recognizer.clear_stale_tracks(current_track_ids)
        self.behavior_reasoner.clear_stale_tracks(current_track_ids)

        # 告警标签独立选取：只要本帧存在"规则确认过的高危标签"且置信度过门槛就报，
        # 不再依赖 main_label 的 argmax——多目标场景里一个高置信 Normal/坐姿会把它压掉
        # （实测 laying 跌倒段 P003 H03 0.8 存在，但 main_label 被判成 Normal 0.92）。
        # 只认"规则确认过的高危标签"，不从 main_label 继承：main_label 是展示用的
        # argmax 结果，PC 噪声（pc_gray/vlm_async 缓存）偶尔会把 H03 顶上来造成误报。
        alert_label, alert_conf = "Normal", 0.0
        for _item in pending_tracks:
            _lbl = str(_item.get("label", ""))
            _cf = float(_item.get("confidence", 0.0))
            if not _lbl.startswith("H") or _cf < self.alert_system.high_risk_min_confidence:
                continue
            _src = label_sources.get(str(_item.get("track_id", "")), "")
            if not (_src.startswith("rule") and not _src.startswith("rule_pending")):
                continue
            if not alert_label.startswith("H") or _cf > alert_conf:
                alert_label, alert_conf = _lbl, _cf
        alert = self.alert_system.build_alert(alert_label, alert_conf, frame_id, timestamp=time.time(),
                                              frame=frame)
        if alert["triggered"]:
            self._submit_clip(alert, frame_id, f"event_{int(frame_id)}.json")
            if not self.alert_adjudicator.submit(alert, frame_id):
                self._adjudicate_submit_dropped += 1

        # PoseClassifier 未加载时，使用旧 VLM 复核路径作为后备
        if not self.action_recognizer.pose_classifier_loaded:
            if self.vlm_reviewer.should_review(main_label, main_confidence):
                review = self.vlm_reviewer.review(
                    {
                        "event_id": f"evt_{frame_id}",
                        "behavior_candidates": [{"type": main_label, "confidence": main_confidence}],
                        "risk_level": RISK_MAP.get(main_label, "LOW"),
                    }
                )
                vlm_review = VLMReview(
                    enabled=True,
                    status=review["review_status"],
                    reviewed_behavior=review["reviewed_behavior"]["type"],
                    review_confidence=review["reviewed_behavior"]["confidence"],
                    reason_tags=review["reason_tags"],
                    latency_ms=review["latency_ms"],
                )
                if alert["triggered"]:
                    alert["final_decision"] = "confirmed" if review["review_status"] == "accepted" else "pending"

        stage_post_ms = (time.perf_counter() - stage_post_t0) * 1000.0
        pipeline_compute_ms = (time.perf_counter() - start) * 1000.0

        scene_narration = self.scene_narrator.consume_pending_push()
        alert_adjudication = self.alert_adjudicator.consume_pending_push()

        # 自然语言规则引擎：闭集映射逐帧匹配（零成本）+ 推送队列
        rule_hits: list[dict] = []
        for item in pending_tracks:
            rule_hits.extend(
                self.nl_rule_engine.match_closed(
                    item["label"], item["confidence"], frame_id,
                    source=item.get("label_source"),
                )
            )
        detections_snapshot = [
            {
                "track_id": item["track_id"],
                "bbox": item["bbox"],
                "label": item["label"],
                "confidence": item["confidence"],
            }
            for item in pending_tracks
        ]
        rule_hits = self.nl_rule_engine.commit_hits(rule_hits, frame_id, detections_snapshot)
        if scene_narration:
            self.nl_rule_engine.note_activities(scene_narration.get("notable_activities", []), frame_id)
            self.novel_behavior_memo.note_narrator_activities(scene_narration.get("notable_activities", []), frame_id)
        rule_alerts = self.nl_rule_engine.consume_pending_pushes()
        # 规则告警接入裁决：旧实现只有主告警进裁决器，规则事件永远停在 pending（WEAKNESS_AUDIT #1）
        for rule_hit in rule_alerts:
            rid = str(rule_hit.get("rule_id", "")).strip()
            if not rid:
                continue
            rule_alert = {
                "type": "nl_rule_hit",
                "rule_id": rid,
                "rule_text": str(rule_hit.get("rule_text", "")),
            }
            rule_event_filename = f"event_{int(frame_id)}_rule_{rid}.json"
            self._submit_clip(rule_alert, frame_id, rule_event_filename)
            if not self.alert_adjudicator.submit(
                rule_alert, frame_id,
                event_filename=rule_event_filename,
            ):
                self._adjudicate_submit_dropped += 1
        ctx["detections"] = detections
        ctx["vlm_review"] = vlm_review
        ctx["main_label"] = main_label
        ctx["main_confidence"] = main_confidence
        ctx["stage_action_ms"] = stage_action_ms
        ctx["stage_post_ms"] = stage_post_ms
        ctx["pipeline_compute_ms"] = pipeline_compute_ms
        ctx["alert"] = alert
        ctx["action_runs"] = action_runs
        ctx["scene_narration"] = scene_narration
        ctx["alert_adjudication"] = alert_adjudication
        ctx["rule_alerts"] = rule_alerts
        ctx["rule_override_count"] = rule_override_count
        ctx["reasoner_source_counts"] = reasoner_source_counts
        ctx["reasoner_rule_tag_counts"] = reasoner_rule_tag_counts
        ctx["h01_h04_mapping"] = h01_h04_mapping
        ctx["action_skip_warmup"] = action_skip_warmup
        ctx["action_skip_track_age"] = action_skip_track_age
        ctx["action_skip_gate"] = action_skip_gate
        ctx["action_skip_cached"] = action_skip_cached
        ctx["action_skip_default"] = action_skip_default
        ctx["temporal_result"] = temporal_result
        ctx["pose_skip_motion_gate"] = pose_skip_motion_gate

    def _assemble_result(self, ctx: dict) -> FrameResult:
        """阶段4：组装 FrameResult（含调试指标）。"""
        frame_id = ctx["frame_id"]
        detections = ctx["detections"]
        alert = ctx["alert"]
        vlm_review = ctx["vlm_review"]
        scene_narration = ctx["scene_narration"]
        alert_adjudication = ctx["alert_adjudication"]
        rule_alerts = ctx["rule_alerts"]
        stage_detector_ms = ctx["stage_detector_ms"]
        stage_pose_ms = ctx["stage_pose_ms"]
        stage_action_ms = ctx["stage_action_ms"]
        stage_post_ms = ctx["stage_post_ms"]
        pipeline_compute_ms = ctx["pipeline_compute_ms"]
        action_runs = ctx["action_runs"]
        pose_skip_motion_gate = ctx["pose_skip_motion_gate"]
        rule_override_count = ctx["rule_override_count"]
        reasoner_source_counts = ctx["reasoner_source_counts"]
        reasoner_rule_tag_counts = ctx["reasoner_rule_tag_counts"]
        h01_h04_mapping = ctx["h01_h04_mapping"]
        action_skip_warmup = ctx["action_skip_warmup"]
        action_skip_track_age = ctx["action_skip_track_age"]
        action_skip_gate = ctx["action_skip_gate"]
        action_skip_cached = ctx["action_skip_cached"]
        action_skip_default = ctx["action_skip_default"]
        temporal_result = ctx["temporal_result"]
        label_sources = ctx.get("label_sources", {})
        self._last_frame_id = frame_id
        result = FrameResult(
            timestamp=datetime.utcnow().isoformat() + "Z",
            frame_id=frame_id,
            detections=detections,
            alert=alert,
            vlm_review=vlm_review,
            scene_narration=scene_narration,
            alert_adjudication=alert_adjudication,
            rule_alerts=rule_alerts,
            metrics={
                "pipeline_latency_ms": int(pipeline_compute_ms),
                "label_sources": label_sources,  # 归因诊断：各 track 最终标签来源
                # 告警门槛诊断：build_alert 的入参（H* 且 conf≥0.55 且连续 3 帧才触发）
                "main_label": str(ctx.get("main_label", "")),
                "main_confidence": round(float(ctx.get("main_confidence", 0.0)), 3),
                "pipeline_stage_ms": {
                    "detector": round(stage_detector_ms, 3),
                    "pose": round(stage_pose_ms, 3),
                    "action": round(stage_action_ms, 3),
                    "post": round(stage_post_ms, 3),
                    "sum_stages": round(stage_detector_ms + stage_pose_ms + stage_action_ms + stage_post_ms, 3),
                    "pipeline_compute": round(pipeline_compute_ms, 3),
                },
                "alert_adjudicate_dropped": int(self._adjudicate_submit_dropped),
                "scene_debug": {
                    "narrator": self.scene_narrator.status(),
                    "adjudicator": self.alert_adjudicator.status(),
                    "nl_rule_engine": self.nl_rule_engine.status(),
                    "vlm_discovery": self.vlm_discovery.status(),
                    "crop_classifier": self.crop_classifier.status(),
                    "geom_strong_reviews": self._geom_strong_reviews,
                    "novel_memo": self.novel_behavior_memo.status(),
                    "prototypes": self.behavior_prototypes.status(),
                    "clip_export": self.clip_exporter.status(),
                },
                "model_load_status": {
                    "detector": self.detector.is_real_model_loaded,
                    "pose": self.pose_estimator.is_real_model_loaded,
                    "action": self.action_recognizer.is_real_model_loaded,
                    "pose_classifier": self.action_recognizer.pose_classifier_loaded,
                },
                "detector_debug": {
                    "used_fallback": self.detector.get_debug_info().get("used_fallback", True),
                    "raw_output_shape": self.detector.get_debug_info().get("raw_output_shape"),
                    "effective_person_boxes": self.detector.get_debug_info().get("effective_person_boxes", 0),
                    "error": self.detector.get_debug_info().get("error", ""),
                },
                "pose_debug": {
                    "used_fallback": self.pose_estimator.get_debug_info().get("used_fallback", True),
                    "raw_output_shape": self.pose_estimator.get_debug_info().get("raw_output_shape"),
                    "num_keypoints": self.pose_estimator.get_debug_info().get("num_keypoints", 0),
                    "error": self.pose_estimator.get_debug_info().get("error", ""),
                    "pose_stride": self.pose_stride,
                    "max_pose_tracks": self.max_pose_tracks,
                    "pose_cache_size": len(self.pose_cache),
                    "pose_motion_gate_iou_threshold": self.pose_motion_gate_iou_threshold,
                    "pose_skip_motion_gate": pose_skip_motion_gate,
                },
                "action_debug": {
                    "used_fallback": self.action_recognizer.get_debug_info().get("used_fallback", True),
                    "pose_classifier_loaded": self.action_recognizer.pose_classifier_loaded,
                    "raw_output_shape": self.action_recognizer.get_debug_info().get("raw_output_shape"),
                    "top1_label": self.action_recognizer.get_debug_info().get("top1_label", "Normal"),
                    "top1_score": self.action_recognizer.get_debug_info().get("top1_score", 0.0),
                    "buffer_len": self.action_recognizer.get_debug_info().get("buffer_len", 0),
                    "error": self.action_recognizer.get_debug_info().get("error", ""),
                    "mapping_mode": self.action_recognizer.get_debug_info().get("mapping_mode", "unknown"),
                    "external_top1_label": self.action_recognizer.get_debug_info().get("external_top1_label", ""),
                    "source_label_count": self.action_recognizer.get_debug_info().get("source_label_count", 0),
                    "action_stride": self.action_stride,
                    "max_action_tracks": self.max_action_tracks,
                    "action_min_track_score": self.action_min_track_score,
                    "action_warmup_frames": self.action_warmup_frames,
                    "action_min_track_age": self.action_min_track_age,
                    "action_runs": action_runs,
                    "action_skip_warmup": action_skip_warmup,
                    "action_skip_track_age": action_skip_track_age,
                    "action_skip_gate": action_skip_gate,
                    "action_skip_cached": action_skip_cached,
                    "action_skip_default": action_skip_default,
                    "rule_override_count": rule_override_count,
                    "reasoner_source_counts": reasoner_source_counts,
                    "reasoner_rule_tag_counts": reasoner_rule_tag_counts,
                    "h01_h04_mapping_summary": {
                        "base_h01": h01_h04_mapping["base_h01"],
                        "base_h04": h01_h04_mapping["base_h04"],
                        "refined_h01": h01_h04_mapping["refined_h01"],
                        "refined_h04": h01_h04_mapping["refined_h04"],
                        "base_h01_avg_conf": round(h01_h04_mapping["base_h01_conf_sum"] / max(1, h01_h04_mapping["base_h01"]), 4),
                        "base_h04_avg_conf": round(h01_h04_mapping["base_h04_conf_sum"] / max(1, h01_h04_mapping["base_h04"]), 4),
                        "refined_h01_avg_conf": round(h01_h04_mapping["refined_h01_conf_sum"] / max(1, h01_h04_mapping["refined_h01"]), 4),
                        "refined_h04_avg_conf": round(h01_h04_mapping["refined_h04_conf_sum"] / max(1, h01_h04_mapping["refined_h04"]), 4),
                    },
                },
                "temporal_adjudicator_debug": temporal_result.get("debug", {}),
            },
        )

        if result.alert.triggered:
            self.alert_system.write_event(result.model_dump())

        return result

    # ── B6: 管道模式接口 ──────────────────────────────────────

    def _pipeline_worker(self, stage_no: int) -> None:
        q_in = {1: self._pipe_q1, 2: self._pipe_q2, 3: self._pipe_q3}[stage_no]
        q_out = {1: self._pipe_q2, 2: self._pipe_q3, 3: None}[stage_no]
        while True:
            ctx = q_in.get()
            if ctx is None:
                return
            if ctx.get("generation") != self._pipe_generation:
                continue  # 上一任务残留帧，丢弃
            try:
                if stage_no == 1:
                    self._stage1_detect(ctx)
                elif stage_no == 2:
                    self._stage2_pose(ctx)
                else:
                    self._stage3_recognize(ctx)
                    result = self._assemble_result(ctx)
                    with self._pipe_results_lock:
                        if ctx.get("generation") == self._pipe_generation:
                            self._pipe_results.append(result)
                    continue
            except Exception as exc:
                # 该帧丢弃（与历史"忙时丢帧"语义一致），不阻塞后续帧
                self._last_pipeline_error = f"stage{stage_no}: {type(exc).__name__}: {exc}"
                continue
            if q_out is not None:
                q_out.put(ctx)

    def submit_frame(self, frame, frame_id: int, orig_frame=None) -> bool:
        """B6 管道模式：投递一帧，立即返回。False = 队列满被丢弃。"""
        ctx = self._make_ctx(frame, frame_id)
        if orig_frame is not None:
            ctx["orig_frame"] = orig_frame
        ctx["generation"] = self._pipe_generation
        try:
            self._pipe_q1.put_nowait(ctx)
            return True
        except Exception:
            self._pipe_dropped += 1
            return False

    def drain_results(self) -> list[FrameResult]:
        """B6 管道模式：取走所有已完成 FrameResult（FIFO，按帧序）。"""
        with self._pipe_results_lock:
            out = list(self._pipe_results)
            self._pipe_results.clear()
        return out

    def pipeline_stats(self) -> dict:
        def _qsize(name):
            q = getattr(self, name, None)
            return q.qsize() if q is not None else 0
        return {
            "enabled": self._pipeline_enabled,
            "dropped": self._pipe_dropped,
            "pending_results": len(self._pipe_results),
            "q1": _qsize("_pipe_q1"),
            "q2": _qsize("_pipe_q2"),
            "q3": _qsize("_pipe_q3"),
            "last_error": self._last_pipeline_error,
        }

