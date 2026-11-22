# 02 · 架构与实现

> 合并自：`ARCH.md`（主体）、`TECHNICAL_REPORT.md` §2–§8、`PRD.md` §2/§5/§6、`models/README.md`
> 最后更新：2026-11-19 ｜ 口径：与当前代码一致（历史文档中 17 点/无 RTSP/VLM 占位等旧表述已修订）
> 相关文档：[01 产品需求](01_PRODUCT.md) ｜ [03 评估与路线图](03_QUALITY_AND_ROADMAP.md) ｜ [04 边缘化](04_EDGE_AIOT.md) ｜ [05 运行手册](05_OPERATIONS.md)

---

## 1. 设计目标与约束

- **推理硬件**：CPU（ONNX Runtime），无需 GPU；边缘版走昇腾 NPU（OM）/CPU 混合；
- **实时性**：目标 < 300ms/帧（1080P 单路）；实测 p50 ~351ms、p95 631ms（bench300，realtime 档）；
- **零样本启动**：预训练模型 + 关键词映射即可运行，不依赖训练数据；
- **零人工标注**：VLM-as-Teacher 生成伪标签训练轻量模型；
- **全本地**：无云端 API，视频不出域；
- **行为类别**：11 类（Normal + H01-H05 + M01-M03 + S01-S02），可经开放词汇扩展。

---

## 2. 总体架构

### 2.1 分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│ 应用接口层  FastAPI REST /api/v1/* ｜ WebSocket /ws/task/* ｜ Streamlit │
└───────────┬──────────────────────────────────────────────────────┘
┌───────────▼──────────────────────────────────────────────────────┐
│ 推理流水线 InferenceEngine                                        │
│  Detector(YOLOv8n) → Tracker(ByteTrack 简化) → PoseEstimator      │
│    (ViTPose WholeBody 133 点)                                     │
│  → ActionRecognizer 双路径                                        │
│      主路径 PoseClassifier(32×133×3 → MLP → BiGRU → 11 类)        │
│      备用 SlowFast + Kinetics-400 关键词映射                       │
│  → 三级决策（采纳 / VLM 灰区 / SlowFast 回退）                     │
│  → 强制 VLM 复核(H03/M01)  → BehaviorReasoner(位置规则)            │
│  → InteractionAnalyzer → TemporalAdjudicator(默认关)               │
│  → AlertSystem → NLAlertRuleEngine → AlertAdjudicator             │
│  → SceneNarrator → SceneIndex                                     │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 模块清单

| 模块 | 文件 | 职责 |
|---|---|---|
| Detector | `src/core/detector.py` | YOLOv8n ONNX 目标检测（FP32/INT8 双档） |
| Tracker | `src/core/tracker.py` | 多目标跟踪（2026-09-27 重写：恒速运动预测 + 两阶段匹配 + 速度门控） |
| PoseEstimator | `src/core/pose_estimator.py` | ViTPose WholeBody ONNX，133 关键点（17 身体 + 6 足 + 68 面 + 42 手） |
| ActionRecognizer | `src/core/action_recognizer.py` | 双路径行为识别 + 每 track 32 帧姿态缓冲（`append_pose` 公开入口） |
| BehaviorReasoner | `src/core/behavior_reasoner.py` | 规则引擎：S01 徘徊、S02 禁区、M01 吸烟姿态；`update_zones()` 热更新 |
| InteractionAnalyzer | `src/core/interaction_analyzer.py` | 多人交互角色（attacker/victim/bystander） |
| TemporalAdjudicator | `src/core/temporal_adjudicator.py` | H03 跌倒二次确认（默认 `enabled: false`） |
| VLMReviewer | `src/core/vlm_reviewer.py` | VLM 复核：离线教师 + 灰区复核（上下文注入）+ 强制复核（person crop） |
| NLAlertRuleEngine | `src/core/nl_rule_engine.py` | 自然语言告警规则：闭集映射 + 开放词汇 VLM 判定 |
| SceneNarrator | `src/core/scene_narrator.py` | 场景叙述：20s 周期 + 滚动摘要 + 任务隔离 |
| AlertAdjudicator | `src/core/alert_adjudicator.py` | 告警异步裁决：描述→裁决两趟，回写 `final_decision` |
| SceneIndex | `src/core/scene_index.py` | 场景知识库：BM25 + bge 向量混合检索 |
| FrameRingBuffer | `src/pipeline/frame_buffer.py` | 环形帧缓冲，叙述/裁决/强制复核共用 |
| InferenceEngine | `src/pipeline/inference_engine.py` | 主引擎：串联、任务隔离、强制复核节流、三阶段流水线 |
| VideoStream | `src/pipeline/video_stream.py` | 本地文件 + RTSP（指数退避重连）；边缘版三档解码 |
| AlertSystem | `src/pipeline/alert_system.py` | 告警生成、事件 JSON 持久化、带标注截图落盘 |
| ClipExporter | `src/pipeline/clip_exporter.py` | 告警证据切片：文件源 remux/重编码 + 异步队列 + 事件 clip_path 回写 |
| VLMRouter | `src/pipeline/vlm_router.py` | VLM 请求并发槽位（默认 4）与降级状态 |
| RestAPI | `src/api/rest_api.py` | RESTful 接口 + 后台任务（train/evaluate/render） |
| WebSocketServer | `src/api/websocket_server.py` | 实时推送 + 历史回放 |
| Evaluator | `src/evaluation/evaluator.py` | 伪真值评估（bbox IoU 匹配） |
| Training | `src/training/pose_classifier.py` | PoseClassifier 模型/数据集/训练循环/ONNX 导出 |
| ConfigLoader / Preflight / Schemas | `src/utils/*` | 配置加载、模型预检与档位解析、Pydantic 模型 |

---

## 3. 感知层

### 3.1 Detector

```
输入: frame (H,W,3) BGR
1. letterbox → 640×640 → normalize → HWC→CHW → batch
2. ONNX Runtime (CPUExecutionProvider, ORT_ENABLE_ALL)
3. sigmoid → conf ≥ 0.35 → NMS → 坐标缩放回原图
输出: [{bbox:[x,y,w,h], score, class_id}]
```

- FP32 / INT8 双档，`active_onnx_path` 切换；
- 预检自动探测 ONNX 输入输出签名，不硬编码 tensor 名；
- `_canvas_buffer/_rgb_buffer/_tensor_buffer` 复用，减少每帧内存分配；
- 模型缺失时优雅降级（`is_real_model_loaded=False`，返回空列表）。

### 3.2 Tracker

2026-09-27 重写版（原纯 IOU 贪心版本碎片化严重，真机 600 帧碎片 22→3，↓7.3×）：

- 恒速运动预测（卡尔曼式位置外推）+ 两阶段匹配（高分框优先，低分框二次关联）+ 速度门控；
- track ID 格式 `P001`；连续 `max_missed` 帧未匹配删除；
- 可选 ReID 增强（`reid_model_path`，默认未启用）；
- `advance_without_detections()` 支持检测跳帧推进。

### 3.3 PoseEstimator

```
person crop → 1.25 倍外扩 + 保持 3:4 宽高比 → resize 192×256
→ ImageNet mean/std 归一化 → ONNX → 关键点缩放回原图
输出: 133 × [x, y, conf]
```

- B 版 FP32（高精度）/ S INT8（实时）双档；
- 姿态运动门控：目标框 IOU 变化小于阈值时复用缓存（`enable_pose_motion_gate`，默认关）；
- 手部点 idx 91–132 是吸烟/手机等精细规则的基础；
- 2026-09 修复：补 mean/std 归一化、外扩仿射、解码分母；`infer_batch` 形状 bug 曾致引擎姿态 100% fallback（已修，fallback 60→0）。

---

## 4. 行为识别层

### 4.1 设计哲学：分层开放行为识别

```
PoseClassifier 主路径(11 类闭集, ~10ms CPU)     ← 高频、低延迟
   ├─ conf ≥ 0.75          直接采纳
   ├─ 0.40 ≤ conf < 0.75   VLM 灰区复核（场景上下文 + 16 条行为序列）
   └─ conf < 0.40          SlowFast 关键词映射回退
强制 VLM 复核（H03/M01, person crop 高清帧）      ← 易混淆类无差别复核
```

用分层架构化解"实时"与"开放"的矛盾：轻量闭集守住毫秒级，大模型只处理灰区与长尾。

### 4.2 PoseClassifier（VLM-as-Teacher 主路径）

**数据管道**（`tools/vlm_label_pipeline.py` + `tools/extract_action_training_data.py`）：

1. 视频逐帧检测跟踪 → 每 60 帧采样 person crop（3 帧时序）；
2. Qwen2.5-VL（vLLM，并发 6）输出 `{label, confidence, reasoning}`；
3. conf ≥ 0.6 过滤，与姿态序列对齐 → NPZ `pose_seq (N, 32, 133, 3)`；
4. 最终 manifest 23,637 样本（分布见 [01 §5.2](01_PRODUCT.md)）。

**训练数据来源分组**（自原 `ARCH.md` §10.4 迁入，历史文档已删除）：

| 来源 | 视频数 | 样本数 | 标注方式 |
|------|--------|--------|---------|
| laying 视频 | 5 | ~1,500 | VLM 伪标签 |
| smoking 视频 | 4 | ~1,100 | VLM 伪标签 |
| whole_process 视频 | 10 | ~5,000 | VLM 伪标签 |
| 手动验证集 | 19 | 326 | 人工标注（仅用于评估，不参与训练） |

**网络结构**：

```
输入 (B, 32, 133, 3)   — (x, y, conf)，坐标按 bbox 中心/对角线归一化
  → 空间编码器: Linear(3→64)→BN→ReLU→Linear(64→128)→BN→ReLU（逐关键点独立）
  → 双向 GRU: input 128×133, hidden 256, 2 层, dropout 0.3
  → 分类头: Linear(512→128)→BN→ReLU→Dropout(0.3)→Linear(128→11)
  → softmax
```

**训练**：加权 CrossEntropy（逆频加权 clip 0.5–10）、AdamW（lr 1e-3，wd 1e-4）、CosineAnnealing、100 epoch、batch 64、按视频分层 15% 验证；导出 ONNX（opset 17，动态 batch）。

> 历史教训：v1 曾因训练/推理分布脱节（无归一化、`augment` 把坐标 clip 到 ±1）导致精度失真；v2 用当前管线重新生成数据并重训，best val 69.88%，生产已部署 v2。

### 4.3 姿态缓冲管理（关键实现细节）

- 每 track `deque(maxlen=32)`；PoseClassifier 需连续 32 帧才推理；
- **主路径修复（2026-08）**：原实现只在 SlowFast 兜底路径写缓冲，主路径长期不满 → 大量 Normal。现公开 `append_pose()`，引擎在 PC 批次推理前对每个候选 track 写入当前姿态；
- **形状污染防护**：空姿态/异常形状直接跳过（原 17 点零数组混入 133 点缓冲 → `np.stack` 异常被静默吞掉 → 回退 Normal）；
- **时间粒度**：推理侧每帧追加（而非每 action 步长追加），与训练一致。

### 4.4 三级融合决策

| 区间 | 决策 | 实现要点 |
|---|---|---|
| ≥ 0.75 | 直接采纳 | 主路径输出 |
| 0.40–0.75 | VLM 灰区复核 | 注入场景滚动摘要 + 该目标近期 16 条行为序列；prompt "先描述动作过程再给结论"，单次链式输出 `description+label+confidence`；**异步提交、回写不阻塞** |
| < 0.40 | SlowFast 回退 | 关键词映射；edge 档可禁用 |

### 4.5 强制 VLM 复核（易混淆类专项）

配置 `force_vlm_review: {classes: [H03, M01], throttle_frames: 25}`：

- **H03**：坐姿/半躺与躺卧姿态高度相似，高置信度误判会绕过灰区 → 不论置信度一律复核；
- **M01**：姿态分类器看不到烟，手近嘴与摸脸不可分 → 同样强制复核；
- **person crop 模式**：从 FrameRingBuffer 按 bbox 裁 3 帧高清 crop（全场景 480px 不够看清手部）；
- **节流**：同目标 25 个处理帧内复用上次结果；VLM 不可用时回退原分类；
- **效果**：跌倒误报 89 → 6 次；smoking 视频 M01 检出 0 → 177 帧（1200 帧验证）。

### 4.6 备用路径：SlowFast 关键词映射

SlowFast ONNX（32 帧 224×224，alpha 4）输出 Kinetics-400 概率 → softmax top1 → 关键词表归并 11 类。映射仅取 top1 且 Kinetics 语义与校园有差距，故只作回退；S01 关键词已移除 `walking/standing/waiting`（正常行为不该映射为徘徊）。

```python
DEFAULT_KEYWORD_RULES = {
  "H01": ["fighting", "wrestling", "boxing", "punch", "kick", ...],
  "H02": ["beating", "slapping", "assault", "attacking", ...],
  "H03": ["fall", "slip", "stumble", "tripping", "collapse", ...],
  "H04": ["jumping", "diving", "cliff", "ledge", "hanging", ...],
  "H05": ["breaking", "smashing", "destroy", "throwing", ...],
  "M01": ["smoking", "cigarette", "lighting", "lighter", ...],
  "M02": ["phone", "calling", "texting", "smartphone", ...],
  "M03": ["covering camera", "blocking lens", "tampering camera", ...],
  "S01": ["wandering", ...],   # 已移除 walking/standing/waiting
  "S02": ["climb fence", "crossing barrier", "intrusion", ...],
}
```

### 4.7 BehaviorReasoner（轻量规则，只保留位置/姿态相关）

VLM-as-Teacher 落地后，肢体动作规则（打架/跌倒/吸烟全量版）已移除，`refine_light()` 只保留：

| 规则 | 判据 |
|---|---|
| **S01 徘徊（pacing 默认唯一通道）** | 45s 滑动窗口内：活动半径 ≤0.35×人体对角线 + 轨迹展开 ≥0.10×对角线（排除站立抖动）+ 路径长/净位移 ≥2.5（来回走而非直线穿过）；置信度按驻留时长爬升、封顶 0.90 |
| **S01 dwell 长时滞留** | ≥300s，默认关闭（`enable_dwell_loiter`） |
| **S02 禁区** | 多边形点在区域内即触发；`forbidden_zones` REST 热更新 |
| **M01 吸烟姿态** | 鼻尖（idx 0）与手部点（idx 91–132）距离连续 10 帧命中；只给 0.72 中等置信度 → 落灰区 → VLM 确认 |
| **M02 手机姿态** | 手部靠近头部 + 低头姿态（`phone_posture_rule`，2026-09 移植） |
| **M03 遮挡** | BBox 面积占比 + 靠近画面中心 |

> S01 语义硬约束：**不得覆盖任何具体行为**（M02/M01/H*）；模型输出的 S01 一律重映射为 Normal（S01 只能由轨迹规则产生）。旧版"90 帧内位移 <80px"曾把站立抖动判为徘徊（592 样本中 301 误报），已按视频时间口径重写。

### 4.8 InteractionAnalyzer

```
输入: bbox + keypoints + peer tracks
1. 上肢活动度 = 手腕-肩部距离 / bbox 对角线
2. 防御姿态 = 下蹲 + 遮挡头部综合分
3. 靠近程度 = 与其他 track 的 bbox 中心距离
规则:
  H02: 防御 ≥0.55 → victim；活动度 ≥0.18 → attacker
  H01: 双方活动度 ≥0.18 → 互殴；一方防御 → victim
  默认 bystander
```

> 状态：role 字段未经任何指标验证，仅展示，不得用于告警决策（见 03 §4）。

### 4.9 TemporalAdjudicator

16 帧滑动窗口，特征 `[h03_ratio, h04_ratio, h02_ratio, h01_ratio, h03_conf_avg, h04_conf_avg, edge_risk_ratio, high_risk_ratio]`，三分类：`direct_fall / risk_then_fall / non_fall`；支持启发式或逻辑回归权重。**默认 `enabled: false`**（`update()` 内快速返回，无计算浪费）；待 H03 数据增强后重新评估价值。

---

## 5. 语义与场景层

### 5.1 VLMReviewer（两阶段使用）

- **阶段一（离线教师）**：3 帧时序 person crop → vLLM → 伪标签；覆盖 19 视频、23,637 样本；
- **阶段二（运行时复核）**：灰区（0.4–0.75）+ 强制类（H03/M01）；输入 3 帧 + person crop + 骨架叠加；输出 `(label, confidence, reason_tags, description)`；
- **运行模式**：vLLM API（吞吐）或 transformers 直载（低延迟单次）；
- **降级**：VLM 不可用 → 返回原分类结果并标记 degraded，不编造结论；
- **`review_max_tokens`**：默认 256（原 128 会截断链式输出，已修）。

### 5.2 NLAlertRuleEngine（开放词汇，赛题核心能力）

```
用户一句话 → ① 规则解析（本地 VLM 纯文本调用，2–4s；失败降级中文关键词表）
             → {matched_classes[], open_description, use_open_vocabulary, confidence_threshold}
           ②a 闭集映射（逐帧零成本）：命中 11 类之一 → 主路径结果直接命中
           ②b 开放词汇判定（VLM）：体系外需求 → 环形缓冲关键帧逐条独立判定
             周期 15s + 叙述器报告异常时加急（间隙 6s）；异步单飞不阻塞
命中 → event_{fid}_rule_{rid}.json → WS 推送 rule_alerts → 墙钟冷却 10s
```

REST CRUD：`POST/GET /api/v1/alert-rules`、`GET/PUT/DELETE /api/v1/alert-rules/{rule_id}`，持久化 `configs/alert_rules.json`。

**四条工程教训（均为实测踩坑，复现成本高）**：
1. 判定 prompt 必须**填空式**（`<true或false>` 占位）；给完整示例 JSON 时 7B 会照抄默认值（命中率曾虚降至 0–16%）；
2. **每规则一次独立小调用**——多规则合并 JSON 数组输出会被 7B 漏答；
3. 冷却用**墙钟时间**而非帧号差（帧号跨任务归零会误压制）；
4. `start_vllm.sh` 内置防重复启动检查——双实例瓜分显存会导致图像编码静默失败。

**开放事件评估**：HMDB51 四类（wave/shake_hands/throw/clap，各 12 段）总体 78%，响应 2–3s。

### 5.3 SceneNarrator

- 触发：每 20s **视频时间**（帧号/fps 换算，与处理节奏解耦）；
- 输入：FrameRingBuffer 均匀采样 4 帧（≤768px）→ vLLM 多图 → JSON `{scene_summary, notable_activities, environment}`；
- 滚动摘要：最近 10 条叙述每新增 5 条压缩为 ≤150 字（原始 JSONL 是真值来源）；
- **检测事实注入（C6）**：引擎把 20s 窗口聚合的逐目标判定注入 prompt"检测模块参考"块；VLM 未提及时代码追加"【检测联动·待核实】"（结构化行，非字符串拼接）；
- **任务隔离**：`begin_task()` 切专属日志 `scene_log_{task_id}.jsonl` 并重置调度；
- 降级：vLLM 失败退避 30s，单飞防堆积；`person_count` 取自跟踪器。

### 5.4 AlertAdjudicator

告警触发后异步"描述 → 裁决"两趟：第一趟对 ±5s 窗口（6 帧）生成事件描述；第二趟结合描述 + 场景上下文判定 `confirmed/rejected/uncertain`，回写事件 JSON。`rejected` 仅标记保留（前端灰显可过滤，保留审计链）。队列有界（`max_pending_alerts` 默认 4），规则告警同样入队裁决。edge 版第一趟改为事实文本合成（不发图）。

### 5.5 SceneIndex

- 索引源：`scene_log*.jsonl` + `event_*.json`（文件指纹变更时增量重建）；
- 检索：BM25Okapi（jieba）主通道 + bge-small-zh-v1.5 向量重排，`0.5×归一化BM25 + 0.5×余弦`，sim ≥0.4 过滤；小语料 IDF 退化时向量兜底；
- 去重：同 task 60s 窗内 sim ≥0.93 的后续叙述不入库；
- 支持 `doc_type` / `task_id` 过滤（过滤在 top_k 截断之前）；
- 模型缺失自动降级纯 BM25。

### 5.6 共享基础件

- `FrameRingBuffer`：每处理帧入队（降采样 480px；原帧复核通道独立缓冲 `max_seconds: 12`），叙述/裁决/强制复核共用；按视频时间窗淘汰，`maxlen` 仅作内存护栏；
- `vlm_client`：vLLM chat 调用 + 宽松 JSON 解析（围栏剥离、数组容错）；
- `vlm_cache`：dHash 9×8 + LRU + TTL 30s，静态场景复核调用减半；
- `VLMRouter`：全局并发槽位（BoundedSemaphore，默认 4），四类调用方统一接入。

---

## 6. 推理流水线

### 6.1 主循环（同步路径，render/离线评估用）

```
1. 帧读取          VideoStream.read() → (ok, frame, frame_id)
2. 检测（步长）     frame_id % detector_stride == 0 ? Detector(frame) : 复用
3. 跟踪            Tracker.update(detections) → ID 切换检测/缓冲清零
4. 姿态（步长）     pose_cache[tid] = PoseEstimator(crop)；否则复用
5. 行为（步长）     每 track append_pose → 缓冲满 32 → PC 推理 → 三级决策
                   → refine_light → 角色分析 → 时域裁决
6. 告警            risk_level == HIGH → AlertSystem.build_alert
7. 推送            FrameResult → WebSocket；事件 JSON 落盘
```

### 6.2 帧跳步与门控

| 模块 | 参数 | 默认（标准） |
|---|---|---|
| 检测 | `detector_stride` | 1（每帧） |
| 姿态 | `pose_stride` | 8 |
| 行为 | `action_stride` | 6 |
| 推送 | `push_stride` | 1 |

- **姿态运动门控**：bbox IOU 与上次 > 阈值 → 复用姿态；
- **检测器运动门控**：画面灰度变化 < 阈值 → 复用上次检测。

### 6.3 三阶段流水线（B6，已启用）

`process()` 拆为四个可复用阶段：`_stage1_detect`（门控+检测/跟踪+C1 清零）、`_stage2_pose`（状态装配+姿态批推理）、`_stage3_recognize`（三级决策+规则+C2 平滑+告警/旁路）、`_assemble_result`。

- **管道路径**：三个常驻工作线程 + `Queue(maxsize=4)` 串联，帧间重叠；`submit_frame()`/`drain_results()`/`pipeline_stats()`；代际计数（generation）实现任务隔离；
- **配置**：`inference.pipeline.{enabled,queue_size}`，当前 `enabled: true`，一键回退串行；
- **实测**：160/160 帧与串行完全一致（0 丢帧），吞吐 1.84 → 4.15 帧/s（2.26×）。

---

## 7. 配置系统

```
configs/
├── model_config.yaml       # 模型路径 + 推理参数 + 档位（profiles）
├── behavior_rules.json     # 规则参数（约 60 项）+ risk_map + alert_policy
├── vlm_config.yaml         # VLM 端点/复核策略（crop_review 段已随边缘代码移除）
├── camera_config.yaml      # 摄像头 + 布控区域持久化
├── alert_rules.json        # 自然语言规则持久化
└── labels/                 # detector/pose/action 标签
```

`behavior_rules.json` 结构：

```json
{
  "risk_map": { "H01": "HIGH", "M01": "MEDIUM", "...": "..." },
  "alert_policy": { "min_consecutive_frames": 3, "cooldown_seconds": 5, "high_risk_min_confidence": 0.55 },
  "heuristics": { "loiter_min_frames": 90, "fall_aspect_ratio": 0.62, "h01_motion_ratio": 0.14, "...": "..." }
}
```

档位通过 `INFERENCE_PROFILE` 切换（参数表见 [01 §4.2](01_PRODUCT.md)）。

---

## 8. 数据模型

```python
class FrameResult(BaseModel):
    timestamp: str
    frame_id: int
    detections: list[Detection]   # track_id, bbox, behavior, pose_keypoints
    alert: Alert
    vlm_review: VLMReview
    metrics: dict                 # 各阶段耗时、FPS、丢帧
    rule_alerts: list             # 自然语言规则命中
    scene_narration: dict | None
    alert_adjudication: dict | None
    decision_traces: list[dict]   # 边缘版：时刻④数据源

class Behavior(BaseModel):
    type: str          # Normal | H01..S02
    confidence: float
    risk_level: str    # LOW | MEDIUM | HIGH
    role: str          # attacker | victim | bystander | unknown

class Alert(BaseModel):
    triggered: bool
    type: str
    source: str
    final_decision: str   # pending | confirmed | rejected
    snapshot_url: str     # 实际截图路径（2026-09-28 修复）
    clip_path: str        # 为空（切片未实现）
```

---

## 9. API 设计

### 9.1 REST 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/analyze/stream` | 启动分析任务（本地文件/RTSP） |
| POST | `/api/v1/analyze/render` | 启动渲染任务（输出带标注 MP4） |
| GET | `/api/v1/tasks/{task_id}` | 任务状态 |
| POST/GET | `/api/v1/evaluate`、`/evaluate/{eval_id}` | 后台评估 |
| POST/GET | `/api/v1/train`、`/train/{train_id}` | 后台训练 |
| GET | `/api/v1/media` | 本地视频/截图 |
| GET | `/api/v1/events` | 历史事件（含 `final_decision`） |
| GET/POST/PUT/DELETE | `/api/v1/alert-rules[/{rule_id}]` | NL 规则 CRUD（热生效） |
| GET/POST/DELETE | `/api/v1/zones[/{zone_id}]` | 布控区域（热更新） |
| GET | `/api/v1/scene/log`、`/scene/status` | 叙述日志/状态 |
| GET | `/api/v1/search`、`/search/status` | 知识库检索（`q/top_k/doc_type/task_id`） |
| GET | `/api/v1/preflight` | 模型预检 |
| POST | `/api/v1/profile` | 切换档位 |
| POST | `/api/v1/vlm/review` | 单事件复核（失败 503） |
| GET | `/api/v1/metrics` | NPU/引擎遥测（边缘版） |
| POST | `/api/v1/narrate` | 点谁说谁（边缘版） |
| POST | `/api/v1/chat` | 和录像聊天（边缘版） |
| GET | `/health` | 健康检查 |

### 9.2 WebSocket

```
端点: /ws/task/{task_id}
消息: FrameResult(JSON)，频率由 push_stride 控制
连接后自动回放 replay_buffer_frames 条历史
```

### 9.3 数据流

```
请求 → FastAPI → InferenceEngine.process_frame()
                        ↓ FrameResult
          ┌─────────────┼─────────────┐
      WebSocket     AlertSystem     HTTP 响应
      (实时推送)    (JSON/截图落盘)
```

### 9.4 消息载荷示例（自原 `PRD.md` §6.2 迁入，历史文档已删除）

> 示例为 PRD v2.1 时期的载荷形态，仅作字段参考；实际字段以 `src/utils/schemas.py` 与实现为准（`snapshot_url`/`clip_path` 的历史假路径问题已修复，见 [03 §3.5](03_QUALITY_AND_ROADMAP.md)、[06 §1](06_GAPS_DESIGN.md)）。

**实时分析请求/响应**（`POST /api/v1/analyze/stream`）：

```json
{
    "stream_url": "data/demo_video.mp4",
    "callback_url": "http://manager-system/alert",
    "duration": 3600
}
```

```json
{
    "task_id": "task_20240311_001",
    "status": "running",
    "endpoint": "ws://agent-server/ws/task_20240311_001"
}
```

**WebSocket 实时推送**（FrameResult）：

```json
{
    "timestamp": "2024-03-11T14:30:15.200Z",
    "frame_id": 4520,
    "detections": [
        {
            "track_id": "P001",
            "bbox": [100, 200, 150, 400],
            "behavior": {
                "type": "H01",
                "confidence": 0.95,
                "risk_level": "HIGH",
                "role": "attacker"
            },
            "pose_keypoints": [["x1", "y1", "c1"], "..."]
        }
    ],
    "alert": {
        "triggered": true,
        "type": "h01_detected",
        "source": "base_model",
        "final_decision": "pending",
        "snapshot_url": "/snapshots/alert_4520.jpg",
        "clip_path": "/clips/alert_4520_10s.mp4"
    },
    "vlm_review": {
        "enabled": true,
        "status": "accepted",
        "reviewed_behavior": "H01",
        "review_confidence": 0.97,
        "reason_tags": ["interaction_pattern", "temporal_consistency"],
        "latency_ms": 45
    },
    "metrics": {
        "pipeline_compute_ms": 156,
        "stage_detector_ms": 42,
        "stage_pose_ms": 35,
        "stage_action_ms": 68,
        "stage_post_ms": 11,
        "fps": 8.2
    }
}
```

**评估响应**（`POST /api/v1/evaluate`）：

```json
{
    "accuracy": 0.90,
    "false_positive_rate": 0.10,
    "false_negative_rate": 0.10,
    "avg_latency_ms": 245,
    "per_class_metrics": {},
    "result_file": "/results/result_campus_test.json"
}
```

**预检响应**（`GET /api/v1/preflight`）：

```json
{
    "profile": "realtime",
    "runtime": {"device": "cpu", "backend": "onnxruntime"},
    "models": {
        "detector": {"status": "ok", "path": "models/detector/yolov8n_int8.onnx"},
        "pose": {"status": "ok", "path": "models/pose/vitpose_wholebody_s_int8.onnx"},
        "action": {"status": "ok", "path": "models/action/slowfast_int8.onnx"}
    },
    "inference": {}
}
```

---

## 10. 关键算法细节

### 10.1 SlowFast 关键词映射

```
logits [1,400] → softmax top1 → source_labels 取类别名
→ 遍历 DEFAULT_KEYWORD_RULES 子串匹配 → 命中即映射
→ 不匹配且 conf < keyword_mapping_threshold → Normal
```

### 10.2 跌倒多维度特征（TemporalAdjudicator）

宽高比（站立 >1.0，倒地 <0.6）、下落速度、运动量、静止时长、头部相对位置、中心点高度、身体水平度；多特征加权投票，连续 6+ 帧满足才触发。

### 10.3 布控区域（S02/H04）

前端截帧 → 鼠标点击多边形顶点 → 命名 + 类型（forbidden/danger）→ 保存 `camera_config.yaml` → `BehaviorReasoner.update_zones()` 热更新，无需重启。该功能激活了原本全空多边形的 S02/H04 规则。

---

## 11. 工程实现细节

### 11.1 任务隔离（`begin_task`）

新任务启动时重置：FrameRingBuffer 清空、SceneNarrator 切任务日志并重置触发帧号、AlertAdjudicator 清待裁决、NLAlertRuleEngine 重置调度、VLM 缓存与平滑历史清空。`run_task`/`run_render_task` 自动调用。

### 11.2 异步与并发

- 灰区 + 强制复核异步提交，回调更新 `action_cache` 与事件 JSON；主循环返回 `pending_vlm` 不阻塞；
- 四个 VLM 消费方（叙述/裁决/规则/复核）共享 `VLMRouter` 并发槽位；
- VLM 缓存（dHash + TTL）静态场景调用减半。

### 11.3 视频接入

`VideoStream` 支持本地文件与 RTSP：断线指数退避 2s→30s，恢复后延迟重置。边缘版的 `source_backend=auto`（cv / ffmpeg 管道 / lavfi）随边缘代码一并归档，见 [04](04_EDGE_AIOT.md)。

### 11.4 前端可视化组件

Streamlit 自定义 HTML 组件实现检测框/骨骼叠加：
- 播放位置跨重跑保持（window 变量 + timeupdate 写回 + 重建恢复）；
- 用户手动操作后自动同步永久退出（避免"暂停被自动恢复"）；
- 悬浮窗口拖拽（setPointerCapture）+ 跨文档搬运（adoptNode）后恢复位置；
- 标题/视频列吸顶（CSS sticky + `:has(iframe)`）；
- JS 崩溃显示红色错误框（消息 + 堆栈），定位不靠猜。

### 11.5 关键 bug 修复记录（工程教训，详见 03）

| 问题 | 根因 | 修复 |
|---|---|---|
| 主路径检不出行为 | `_append_pose` 只在兜底路径调用 | 公开 `append_pose()`，主路径推理前写入 |
| 17/133 形状污染 | 空姿态写 17 点零数组 | 空姿态/异常形状跳过 |
| S01 过触发（301/592） | "90 帧位移 <80px" 判据 | 视频时间口径重写 |
| VLM 命中率虚低 | 完整示例 JSON 被照抄 | 填空式 prompt + 回显检测重试 |
| 多规则漏答 | 7B 对 JSON 数组偷懒 | 每规则独立调用 |
| 双 vLLM OOM | 双实例瓜分显存 | 启动脚本防重复 + `--dtype half` |
| 跨任务叙述空白 | 帧号跨任务残留 | `begin_task()` 统一重置 |
| 环形缓冲长期空 | 存储/处理步长互质 | 每处理帧入队 |
| 告警截图全失效 | 局部变量 NameError + 假路径 | 触发点落盘 + media 端点 |
| Prompt 花括号 KeyError | JSON 示例未双写 | `{{ }}` 转义（两次踩中） |

---

## 12. 部署架构

### 12.1 进程模型

```
uvicorn (run.py)
  └── FastAPI App
        ├── REST 处理
        ├── WebSocket 协程
        └── InferenceEngine（同步/三阶段管道）

streamlit (demo/web_interface/app.py)   # 前端 + WS 客户端
vLLM / llama-server (:8001)             # 本地大模型
```

### 12.2 服务与端口

| 服务 | 端口 | 说明 |
|---|---|---|
| vLLM（服务器）/ llama-server（盒） | 8001 | OpenAI 兼容，消费方零改动 |
| FastAPI | 8000 | REST + WS |
| Streamlit | 8501 | 看板 |
| Mosquitto（边缘） | 1883 | MQTT 端云联动 |

远程访问：`ssh -N -L 8501:localhost:8501 -L 8000:localhost:8000 用户@服务器`。盒上用 systemd 三单元（`csa-vlm → csa-engine`，`Restart=always`），见 [05 §7](05_OPERATIONS.md)。

### 12.3 模型文件布局

```
models/
├── detector/  yolov8n.pt / .onnx / _int8.onnx（边缘 .om）
├── pose/      vitpose_wholebody_{s,b}.pth / .onnx / _s_int8.onnx
├── action/    slowfast.pth / .onnx / _int8.onnx
├── action_v2/ pose_classifier_best.pth + pose_classifier.onnx（生产）
├── crop/      crop_v1.onnx（可选：服务器侧 M01 确认钥匙；当前需按监控域重训）
├── embedding/ bge-small-zh-v1.5
└── tracker/   reid.onnx（可选，未启用）
```

---

## 13. 开发规范

- Python 3.10+，type hints（`from __future__ import annotations`），Pydantic v2；
- 配置驱动：所有可调参数进 `configs/`，禁止硬编码路径/阈值；
- 每个核心模块：`config: dict` 构造参数 + `is_real_model_loaded` + `last_debug`；
- **优雅降级**：模型/服务缺失不阻塞启动，返回空结果或默认值并标记；
- 异常在模块内捕获，不向上传播；视频流断开 `read()` 返回 `(False, None, frame_id)`；
- 新模块配冒烟脚本；改动决策逻辑必须跑 golden 回归（`tools/test_golden_regression.py`）。

---

**版本**：整合版 v1.0（源自 ARCH v1.0 / TECHNICAL_REPORT v1.0 及 2026-09~11 增量修复）
**更新日期**：2026-11-19
