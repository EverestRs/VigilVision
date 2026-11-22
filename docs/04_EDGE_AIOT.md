# 04 · 边缘化与参赛（Atlas 单盒 · AIoT）

> 合并自：`EDGE_MIGRATION.md`（v3.0）、`AIoT_COMPETITION.md`（v3.0）、`edge_migration/README.md` + `01/02/03` + `FIELD_CHECKLIST.md` + `FIELD_READOUT_20261105.md`、`deploy/README.md`
> 最后更新：2026-11-19 ｜ 口径：方案为 v3.0（算法主体 + 最小硬件壳），实测截至 2026-11-05
> **实现状态（2026-11-19）**：边缘迁移已验证完成，边缘**运行时代码**（`edge_migration/` 包、`deploy/`、OM 后端、crop 仲裁接线、MQTT、语言层端点）已从仓库移除；crop **训练工具与数据集保留**（`tools/train_crop_classifier.py`、`tools/tune_crop_thresholds.py`、`results/crop_dataset/`），供服务器侧 M01 确认钥匙路线复用。本文档作为**方案设计与盒上实测记录存档**，文中路径/命令不再指向仓库内文件。
> 相关文档：[01 产品需求](01_PRODUCT.md) ｜ [02 架构与实现](02_ARCHITECTURE.md) ｜ [03 评估与路线图](03_QUALITY_AND_ROADMAP.md) ｜ [05 运行手册](05_OPERATIONS.md)

---

## 1. 定位与价值主张

### 1.1 v3.0 核心转向

目标从"工程完备的边缘移植"改为**"算法主体 + 最小硬件壳"**：砍掉为稳健性服务、但 5 分钟演示看不见的一切投入（多路并发 / 24h 烤机 / INT8 精度全表 / ESP32 执行链 / 内存苦战），把所有工时转投**算法创新性与可展示性**。

**支点：把 VLM 从"质量裁判"改成"语言接口"**。v2.x 让边缘 VLM 干它不擅长的细粒度视觉判定（灰区复核、开放词汇看图），押着两座悬崖：质量悬崖（7B 对 <150px 已 50/50）+ 工具链悬崖（多模态在昇腾 310B 支持度未知）。v3.0 改为：

```
视觉判定 → 确定性感知层（NPU OM + crop 多分类器 + 规则引擎）   毫秒级、可复现、无悬崖
语言/交互 → 微型文本 LLM（Qwen2.5-0.5B 级，纯文本，非多模态）  解析指令/合成旁白/答问/讲解
```

三个后果：① 命门出清（纯文本小模型在昇腾有成熟路径，多模态降为可选 stretch）；② 4GB 从苦战变从容（1.5B Q4 ≈1GB，不再需要交替独占加载）；③ 叙事更硬——"不是把大模型硬塞进边缘盒，而是让语言只做语言、确定性小模型守住视觉"。

### 1.2 一句话定位

**慧眼智警·边缘认知盒**——单颗国产昇腾 NPU（8 TOPS、≤20W、全本地）上的**可对话、可布控、可自述思维**的视觉安全 AIoT 系统：确定性感知层守住毫秒级判定，微型语言层负责"听懂人话、说出人话"，整条认知链路在演示屏上实时可见。

### 1.3 价值宣言（评委视角三句话）

1. 不做"把大模型硬塞进开发板"——做的是**边缘分层认知架构**：语言模型只做语言，确定性小模型守住视觉；
2. 不做"跑分自证的算法 demo"——把 **AI 的决策过程做成演出**：三级决策、复核翻案、置信度流转实时字幕化；
3. 单盒、无云、全国产、隐私不出域——**资源受限不是妥协，是设计**。

### 1.4 诚实交底（评分取舍）

- **损失**：完整性 10% 的实物外壳/PCB 分、创新性中"硬件形态/传感组合"、技术性中"软硬件设计复杂度"的硬件半边；
- **保住/增益**：技术性 35% 的"边缘 AI 部署优化（内存/速度/功耗）"完全保住（OM + VDEC + 单盒 + 文本 LLM + npu-smi 功耗）；创新性靠算法（零标注蒸馏、一句话布控、可解释复核链、场景记忆问答）拉满；
- **合规底线**：摄像头 = 数据采集、MQTT = 网络通信、单盒 = 边缘 AI 推理，三能力齐，仍是合法 AIoT 作品，只是轻壳。

### 1.5 关键时间事实

省赛作品提交截止 **2026-11-18**（v3.0 方案按 7.5 周倒排）。**当前日期 2026-11-19，倒排计划已到期**；实际执行状态以 §8/§9 为准——代码任务已全部完成，盒上实测在 11-05 完成首轮，OM 加载与帧率达标仍待解决。

---

## 2. 四层算法架构（最小合规壳）

```
┌── 边缘推理盒 Atlas 200I DK A2（单盒 · ≤20W · 全本地 · 无云）──────────┐
│ L1 确定性感知  RTSP/USB → VDEC 硬解 → YOLOv8n OM@NPU → ByteTrack      │
│                → ViTPose(OM/CPU) 133 点 → PoseClassifier              │
│                → Crop 多分类器(M01/H03/挥手/鼓掌/打电话…)             │
│                → 行为规则(S01 徘徊/S02 禁区)                          │
│ L2 分层决策    三级置信度分流（无 VLM 版）：高→采纳 / 灰区→crop 仲裁   │
│                / 低→SlowFast 兜底（可禁）；C2 时序平滑                 │
│ L3 语言交互    Qwen2.5-0.5B(纯文本)：一句话布控解析·感知事实→旁白合成  │
│                ·场景记忆问答·规则执行语义                             │
│ L4 可解释可视化 复核链路字幕流（0.61 徘徊→crop 判非吸烟→翻案→Normal）  │
│                点谁说谁旁白·底牌数据(fps/内存/功耗)                    │
│ ── 服务 ── FastAPI+WebSocket :8000 ｜ Mosquitto ｜ Streamlit 看板      │
└────────────────────────────────────────────────────────────────────────┘
        MQTT(网络通信) → 笔记本/手机浏览器看板(IoT 应用)
```

AIoT 三能力映射：数据采集 = 摄像头图像传感器（H.264 RTSP 全链路）；网络通信 = MQTT（遥测/事件上行、规则下行、LWT 在线）；边缘 AI 推理 = 单盒四层。

---

## 3. 硬件事实与 M0 可行性

### 3.1 硬件参数

| 项 | Atlas 200I DK A2（310B） |
|---|---|
| AI 算力 | 8 TOPS INT8 / 4 TFLOPS FP16 |
| CPU | 4× Cortex-A55 弱核（感知编排瓶颈，靠 VDEC + 步长缓解） |
| 内存 | 4GB / 8GB 两版；**实测为 4GB（310B4，3.4Gi usable，净余 2.2Gi）** |
| 视频 | VDEC/VPU 硬解（H.264/265）；实跑以 cv 软解为预演档 |
| 软件栈 | CANN 6.2.T2.0.B133（6.2.RC2）、ATC(ONNX→OM)、llm.cpp / llama-server |

### 3.2 Go/No-Go 验证线（G1–G6）

| 门 | 内容 | 通过线 | 实测（2026-11-05） | 档位决定 |
|---|---|---|---|---|
| G1 | 内存版本 | 记录 4/8GB | **4GB**（RSS 1.11G，净余 2.2Gi） | 1.5B 可常驻但不建议 → **0.5B 档** |
| G2 | 解码预算 | 解码 CPU <1.5 核 | **cv 过**（1.38 核，58.6fps，300/300 帧）；ffmpeg 0.06 核* | `source_backend=cv`（正式 RTSP 待真流复测） |
| G3 | YOLOv8n ATC→OM | OM 可推理；IoU≥0.95 | **OM 不上，退 ONNX**（yolo+crop 转换成功；pose int8 因 QLinear opset11 无 parser 失败；pyACL 约定错配） | 退 int8 ONNX |
| G4 | crop 时延 p95 | ≤20ms | **41.9ms（p50 25.3）❌** | 224 + `throttle_frames` 24 |
| G5 | 文本 LLM | 解析 ≤2s，JSON ≥95% | valid 96–100%，p50 **3.9–8.6s ❌**；1.5B 仅 2.6 tok/s 否决 | **0.5B 定案 + 异步 + demo_pack 兜底**；口径改 ≤10s |
| G6 | GRU 时延 p95 | ≤50ms | **174.9ms（p50 110.7）❌** | `action_stride` 8→12 |
| bench | 帧率/延迟 | ≥2fps，P50 ≤400ms | 0.94 → 1.80 → **1.82fps / 546ms ❌** | **触发会商线 §3-2** |

\* ffmpeg 档 CPU 记账只采本进程不含子进程，口径偏低。

### 3.3 内存预算（算法主体版）

| 组件 | 预估 |
|---|---|
| OS + CANN 驱动/运行时 | ~1.5GB |
| 感知 OM（YOLOv8n + ViTPose）@NPU | ~0.8GB |
| Python 引擎 + 帧缓冲 + FastAPI + Mosquitto | ~1.2GB |
| 文本 LLM 0.5–1.5B INT4 常驻 + KV(2k) | ~0.5–1.2GB |
| Crop 分类器 + bge-small + BM25 | ~0.5GB |
| **合计** | **~4.5–5.2GB → 4GB 偏紧（0.5B 可跑），8GB 从容** |

---

## 4. 模块设计与改造

### 4.1 代码库现状映射（文件级）

| 现有文件 | 边缘改造 | 动作 |
|---|---|---|
| `src/pipeline/video_stream.py` | `source_backend` 三档（cv / ffmpeg 管道 / auto） | 改（已完成） |
| `src/core/detector.py` / `pose_estimator.py` | 接入推理后端抽象（ORT/OM 双后端） | 改（已完成） |
| `src/core/action_recognizer.py` | GRU 保留；SlowFast 可禁 | 不改 |
| `src/core/behavior_reasoner.py` | S01/S02/M01 规则 | 不改 |
| `src/pipeline/inference_engine.py` | `decision_mode=edge` + `decision_traces` | 改（已完成） |
| `src/core/vlm_reviewer.py` | server 档保留；edge 由 crop 仲裁替代 | 不动 |
| `src/core/nl_rule_engine.py` | 文本 LLM 天然兼容；增 MQTT 下行入口 | 小改（已完成） |
| `src/core/scene_narrator.py` | `narration_mode: facts / images` 分支 | 改（已完成） |
| `src/core/scene_index.py` | 修 doc_type 过滤顺序；作为问答底座 | 小改 |
| `src/core/alert_adjudicator.py` | `submit()` 队列化；第一趟事实化 | 改（已完成） |
| `src/pipeline/alert_system.py` | MQTT 发布回调 | 小改 |
| `src/utils/schemas.py` | `FrameResult.decision_traces` | 改（已完成） |
| `src/api/rest_api.py` | `/narrate` `/chat` `/metrics` `/demo/*` | 改（已完成） |
| `demo/web_interface/app.py` | 认知视图/角标/点击旁白/问答面板 | 改（已完成） |
| `configs/model_config.yaml` | 新增 `edge` 档位 | 已完成 |
| `configs/vlm_config.yaml` | `crop_review` 段 + edge 分支 | 已完成 |

**直接复用资产**：三级决策框架、tracker/规则引擎、帧缓冲、VLM 缓存（throttle 模式照搬给 crop）、VLMRouter 降级语义、`begin_task` 隔离、评估体系、golden 回归范式、冒烟脚本范式。

### 4.2 新增模块

```
src/core/inference_backend.py    推理后端抽象（ORT / AscendOM），按扩展名分流
src/core/crop_classifier.py      Crop 多分类仲裁器（视觉判定确定性核心）
src/core/fact_compiler.py        感知事实编译器（场景/目标/事件三视图）
src/core/track_narrator.py       按目标旁白合成（点谁说谁 + 反幻觉检查）
src/core/decision_trace.py       决策链数据结构（时刻④）
src/pipeline/mqtt_bridge.py      MQTT 上行/下行桥（dry-run 降级）
src/utils/npu_telemetry.py       npu-smi 功耗/温度/占用（x86 返回 None）
tools/train_crop_classifier.py   crop 训练/导出（x86）
tools/export_om_atlas.sh         ATC 批量导出 + bench
scripts/start_llm_atlas.sh       盒上文本 LLM 服务
deploy/*.service                 systemd 单元
models/crop/                     crop 产物
edge_migration/scripts/g*.py|sh  M0 探针
requirements-aarch64.txt         盒上依赖
```

### 4.3 关键设计要点

**推理后端抽象**：`create_backend(path)` 按 `.om` / `.onnx` 分流；`is_real` 对齐 `is_real_model_loaded`；OM 失败自动回退同源 ONNX 并标记。协议：`input_shape()` / `run(tensor)`。

**Crop 仲裁器**（算法承重件）：MobileNetV3-Small，类集 `{Normal, M01, M02, H03, Wave, Throw, Handshake}`（演示类最终由 Clap 调整为 Handshake，三处类集定义同步）；letterbox 224；结果缓存 + throttle。职责从"H03/M01 主闸"升级为**全部视觉判定的确定性核心**。

**edge 决策模式**（兼容护栏：默认 `server`，行为完全不变）：

1. 灰区：`review_clip`（VLM 看图）→ **crop 仲裁**（一致→上调采纳；矛盾→以 crop 为准降档 Normal，置信度取均值）；
2. 强制复核：`force_crop_classes: [H03, M01, M02]`，throttle + 缓存沿用；
3. 兜底：SlowFast 禁用，低置信度直接 Normal 并记 trace；
4. trace：每 track-帧记录 `steps`（pose_classifier → crop_arbiter → final → rule_hit）+ `latency_ms`，只由真实决策路径写入。

**决策语义参数**（有 parity 数据背书）：`normal_rebate_conf=0.75`（弱证据不翻案）、演示类阈值 0.80–0.85、`arbiter_classes=[M01,M02,H03]`、`no_rebate_classes=[H03]`（Normal 不得否决 H03，含缓存写入路径）。

**事实编译器 + 语言层**：`fact_compiler` 产出场景/目标/事件三视图结构化事实；`track_narrator` prompt 明令"只可复述事实，禁止推断画面内容"；`/api/v1/chat` = SceneIndex top5 + 时间窗证据 + LLM 合成。1.5B/0.5B 更换**零代码改动**（OpenAI 兼容端点）。

**MQTT**：

| Topic | 方向 | 载荷 |
|---|---|---|
| `csa/{box}/status`（QoS0, 1s） | 上 | fps/内存/NPU/功耗/温度/llm_online/source |
| `csa/{box}/event`（QoS1） | 上 | event JSON 摘要 + trace_digest + snapshot_url |
| `csa/{box}/rule/set`（QoS1, retained） | 下 | 布控热更新（与 REST 同一函数） |
| `csa/{box}/rule/ack` | 上 | 规则生效回执（演示时延证明） |

LWT offline；断网缓冲 100 条重发；无 broker 自动 dry-run 降级。

**edge 档位**（`model_config.yaml`）：`decision_mode: edge`、`source_resolution: [1280,720]`、`max_fps: 6`、`pose_stride: 12`、`action_stride: 12`（盒上实测后由 8 改）、`detector_stride: 4`、`max_*_tracks: 3`、`detector_path: models/detector/yolov8n_int8.om`（失败回退 int8 onnx）、`slowfast_enabled: false`。

---

## 5. 里程碑与任务分解

### 5.1 里程碑总览

```
M0(W1) 硬件探针    M1(W2) 感知上盒+crop   M2(W3) 语言层·时刻①②
   │                   │                     │
M3(W4) 时刻③④+连排  M4(W5) 端云联动+评估   M5(W6-W8) 装配/拍摄/文档/提交
```

### 5.2 W1–W8 主任务与出口条件

| 周 | 里程碑 | 核心任务 | 出口条件 |
|---|---|---|---|
| W1 | M0 探针 | G1–G6 + 报名 + 工程基线 + crop 数据重组 | 探针 JSON ≥5 份 + 三条决策记录 |
| W2 | M1 感知上盒 | 后端抽象 / 解码三档 / edge 档 / bench / crop v1 / edge 决策 | 单路 ≥2fps、P50 ≤400ms；parity ≥90% |
| W3 | M2 语言层 | 事实编译器 / 叙述 facts / narrate / 一句话布控 + MQTT / 裁决队列化 | 时刻①② 盒上连排（录屏） |
| W4 | M3 时刻③④ | chat / 认知视图 / 遥测角标 / MQTT / 演示类补全 / 连排 #1 | 每时刻成功率 ≥8/10 |
| W5 | M4 端云联动 | systemd / 评估回归 / 假告警对照 / 证据表 / 连排 #2 | 连排 ≥9/10；指标达标 |
| W6 | 装配拍摄 | 轻壳 / 视频 ≥3 遍 / B 站 / 8h 烤机 / demo 开关 | 粗剪 6–8 分钟 |
| W7 | 文档整备 | 7 项材料 / 仓库 / BOM / 连排 #3 | 材料成稿 |
| W8 | 提交缓冲 | 打包（ZIP ≤200MB）→ 缴费 → 提交 → 回执 | — |

**关键路径**：T1.4 → T2.1 → T2.6 → T3.4 → T5.2（OM → 决策 → 布控 → 评估证据）。连续两周出口未达即触发砍单序。

### 5.3 风险触发动作（预案即代码）

| 触发 | 动作 | 成本 |
|---|---|---|
| G3 OM 失败 | profile 退 `yolov8n_int8.onnx`，stride +2 | 一行配置 |
| G2 解码超预算 | ffmpeg → 640×480 子码流 | 一行配置/采购 |
| G5 JSON 合格率 <95% | 解析修复器 + prompt 收紧 ≤5 字段 | 0.5d |
| crop 某演示类 P<0.5 | 该类退出演示集，文档标注支持范围 | 0d |
| 时刻④ trace 太噪 | `trace_whitelist` 只显示灰区及以上 | 配置 |
| 文本 LLM 不可用 | 关键词表 + 事实模板直出旁白 | 已内置 |
| 现场翻车 | `demo: true` 剧本（预置规则/旁白/问答） | 0.5d |
| 时间塌方 | 砍单序：stretch → 演示类扩展 → 角标功耗 → 装配精度；**绝不砍** edge 决策/时刻①②④/评估 | — |

---

## 6. 验证与测试

### 6.1 M0 探针规约

单文件自包含、`--json-out`、非零退出码即失败；输出统一到 `results/edge_m0/`。

| 脚本 | 指标 | 通过线 |
|---|---|---|
| `g1_mem.sh` | MemTotal | 4G/8G 记录 |
| `g2_decode_bench.py` | 解码 CPU、ms/帧、丢帧（cv vs ffmpeg × 主/子码流） | <1.5 核 |
| `g3_atc_om_bench.sh` | mAP@50、ms/帧、RSS、ORT vs OM 输出差 | 可推理；IoU≥0.95；关键点 L2<2px |
| `g4_crop_latency_bench.py` | crop ORT/OM 各 100 次 | ≤20ms |
| `g5_llm_smoke.py` | tok/s、首 token、四类 prompt ×25 JSON 合格率 | 解析 ≤10s（盒上口径）；≥95% |
| `g6_gru_bench.py` | GRU P50/P95 | ≤50ms |
| `env_probe.sh` | 版本快照 | 写 `ENV_LOCK.md` |

### 6.2 金标回归与 parity

- `tools/test_golden_regression.py`：200 帧/460 检测逐帧基线，conf ±0.02 / bbox ±2px 容差，capture→check 双模式；
- `tools/test_edge_decision_parity.py`：两遍共用 edge 档框架（同步长、同禁 SlowFast），唯一差异 = 仲裁器（server 7B VLM vs edge crop）；681 个判定逐帧对齐，方向一致率 ≥90%；这是"edge 决策是否伤精度"的唯一裁判。

### 6.3 五时刻验收脚本

| 时刻 | 脚本 | 断言 |
|---|---|---|
| ① 一句话布控 | `test_moment1_deploy.py` | 三条口令解析 ≤2s；命中 ≤5s；WS 卡片带 trace |
| ② 点谁说谁 | `test_moment2_narrate.py` | ≤1s 首响应；反幻觉抽查 10/10 |
| ③ 和录像聊天 | `test_moment3_chat.py` | ≤10s 回答；evidence 文件存在、时间戳在窗口内 |
| ④ 认知视图 | `test_moment4_trace.py` | trace 端到端 ≤1s；灰区含 `crop_arbiter`；schema 合法 |
| ⑤ 底牌角标 | `test_moment5_metrics.py` | status 1s；`power_w` 非 null；看板 ≤2s |

### 6.4 证据协议

数字必须来自脚本产物（可追溯文件路径），文档引用附 `results/edge_test/…json` 文件名；禁止手工填数。证据覆盖：帧率/延迟、量化对比（FP32-ORT vs INT8-OM）、解码收益、端侧 LLM（tok/s/合格率）、功耗/内存三态、行为精度、端云时延。

### 6.5 评估回归基线（服务器基线 → 单盒验收线）

> 自原 `EDGE_MIGRATION.md` §7 迁入（历史文档已删除）。

| 指标 | 服务器基线 | 单盒验收线 |
|---|---|---|
| 主路径准确率 | 85.7%/77.7% | 不降（同权重换 OM） |
| 感知流水线 | 363ms/2.8fps | ≤400ms / 2–5fps |
| 跌倒误报 | 89→6（VLM） | ≤15（**crop 多分类器主闸**） |
| 吸烟/行为类检出 | 280/600 | ≥220/600（crop） |
| 开放事件命中 | 78% | crop 演示类可复现 / 8GB stretch VL ≥50% |
| **演示时刻**（新增） | — | ①布控≤2s解析 ②点谁说谁≤1s旁白 ③问答检索命中 ④复核字幕流实时 ⑤底牌角标常显 |

---

## 7. 实现进展汇总（13 批，2026-10-01 ~ 11-05）

| 批次 | 交付 | 验证 |
|---|---|---|
| 首落地 | 后端抽象、crop、事实编译、点谁说谁、决策 trace、遥测、MQTT、aarch64 依赖、selftest | 28/28 PASS；g2 720p 32.8fps/0.1 核；g4 玩具 1.2ms；g6 真模型 23ms；g5 7B 基线 100% 合法 |
| 第二批 | detector/pose `.om` 分支、edge 决策模式、`decision_traces`、裁决队列 bug 修复、edge 档/配置 | smoking 240 帧 PASS（20.6fps，p50 35.6ms）；corner_3 60 帧 PASS_NO_TRACE |
| 第三批 | crop 真模型训练（build/train/export/selftest） | val_acc 92.4%；M01 99.4%/M02 97.1%/H03 100%；ONNX parity 0；g4 3.97ms |
| 第四批 | parity 测试 | **93.0% PASS**（v1 假 PASS → v2 86.5% → v3 96.5% → v4 93.0%）；残余 48 差异为模型层 |
| 第五批 | 叙述 facts 模式、`/narrate` `/chat`、MQTT 接线、引擎支撑 | 语言层冒烟 7/7 |
| 第六批 | 前端五时刻呈现（布控卡/点击旁白/问答/认知视图/角标） | JS esprima 语法检查通过；TestClient 5/5 |
| 第七批 | 解码三档、裁决事实化、demo_pack、systemd、存储 GC、ATC 脚本、bench、LLM 回归、阈值调优 | bench 22.6fps（x86 降级口径）；crop 阈值实跑 |
| 11-05 | 盒上首日实测 + 修正回流（见 §8） | 见 §8 |

**parity 迭代中的真实修复链**（数据驱动）：

| 轮次 | 一致率 | 归因与修复 |
|---|---|---|
| v1 | 100%"一致" | **假 PASS**：server 遍未切 `decision_mode`，两遍都跑 crop |
| v2 | 86.5% FAIL | Normal→Throw/M01→Throw 84 个（演示类域差）+ H03→Normal 24 |
| v3 | 96.5% PASS | `normal_rebate_conf` + 演示类阈值 0.80–0.85 + `arbiter_classes` |
| v4 | **93.0% PASS** | `no_rebate_classes: [H03]` 封堵 Normal 否决 H03 的通道 |

剩余 48 差异（模型层，如实记录）：Normal→M01 24（crop 监控域 M01 精度偏置）；H03→Normal 24（教师偏差致 crop 对跌倒倾向 Normal）。修复路径 = crop 正式重训（监控域困难负样本 + laying 重标），非语义层可解。

**关键踩坑**：`FACTS_NARRATE_PROMPT` JSON 示例花括号未双写 → `.format()` KeyError → 叙述线程退避静默吞掉（与 g5 `PARSE_PROMPT` 同款，第二次踩中）→ 规范：所有含 JSON 示例的 prompt 一律 `{{ }}` 转义或统一 replace 注入，建议 `vlm_client.safe_format` 收口；JS/HTML 插入后必须跑渲染 + esprima 检查。

---

## 8. 盒上首日实测与修正（2026-11-05）

**环境**：davinci-mini，310B4，4GB，CANN 6.2.T2.0.B133，Ubuntu 22.04 aarch64；全天测量在 `sddm-greeter` 常驻 0.55 核环境下进行（无 sudo 未停）。

### 8.1 判定结果

| 项 | 数字 | 决定 |
|---|---|---|
| G1 | 4GB（3.4Gi usable） | 1.5B 可常驻但不建议 → **0.5B** |
| G2 | cv 过（1.38 核，58.6fps，300/300） | `source_backend=cv`（真 RTSP 待复测） |
| G3 | fails=3→实为 1（`.om.om` 命名 bug 已修） | **OM 不上，退 ONNX** |
| G4 | p95 41.9ms（p50 25.3）❌ | 224 + throttle 24 |
| G5 | valid 96–100%，p50 3.9–8.6s ❌ | 1.5B 否决（2.6 tok/s）；0.5B 定案 + 异步 + demo_pack |
| G6 | p95 174.9ms（p50 110.7）❌ | `action_stride` 8→12 |
| bench | 0.94 → 1.80 → 1.82fps / 546ms ❌ | **触发会商线** |

### 8.2 两条会商线

- **§3-1（G2 三档全超）未触发**——cv 过线（文件档预演，真 RTSP 待复测）；
- **§3-2（帧率 <2fps）已触发**。关键事实：① 检查单给的 `frame_stride` **不进引擎处理路径**（检测节拍由 `detector_stride`+门控控制），"stride 加到最激进"从未真正生效；② 盒上实际杠杆已打满（int8 回退 1015→513ms/帧 + detector_stride 4）；③ **真正解法是 NPU OM 跑通**（预期 <20ms/帧 → fps>4）。

### 8.3 会商项与处理

- **A. pyACL 适配（阻塞帧率达标）**：CANN 6.2.RC2 的 `acl.so` 为旧式约定（`load_from_file→(model_id, ret)`、desc 直接给指针、`get_input_dims/get_output_dims`）；代码用新式名 → 永远误判加载失败；按 model_id 传参给 desc 会 segfault。**已实施**：`AscendOMBackend` 重写为约定自适应 shim（`_pair`/`_rc`/`_normalize_dims` 归一 + 函数名探测），新增 `g3_pyacl_probe.py` 子进程隔离探针把盒上约定钉成 JSON。**下次到场第一步**跑探针 → 重跑 export/bench（预期 1.8→4+）。
- **B. 演示口径**：若 A 短期不解决，按 1.5–1.8fps 定档（`max_fps=2`、拉长回放窗口），或 detector 改 416×416 重训/重导。
- **C. pose OM**：QLinear* opset11 不被 CANN 6.2 atc 支持；需转 FP 版（约 30–60 分钟）或 Q/DQ 格式。
- **D. 修正清单（D1–D8）**：export 脚本 `--log` 与 `.om.om` 双后缀；G4 160 输入对静态模型无效；ffmpeg 核数口径；g5 SLOW/UNREACHABLE 分流；requirements 补 6 项（torch/torchvision/transformers/jieba/rank_bm25/onnx）；start_llm `--model-name` 移除；edge detector 路径改指 `yolov8n_int8.om`（回退 int8 而非 FP32）；`.sha256` 伴生文件单独传。
- **E. atc 环境**：TBE 依赖需装在系统 python3.10（非 conda），已装 numpy/decorator/psutil/protobuf/sympy/scipy/six/attrs。

### 8.4 盒上已完成部署物

`~/.conda/envs/csa`（py3.10 + requirements-aarch64 + D5 补齐）、llama-server（源码编译）、GGUF 1.5B/0.5B q4_0、mosquitto debs、`yolov8n.om` + `crop_v1.om`、sudo 服务安装清单。

**待用户三件事**：① 执行 sudo 段（apt 镜像在线可达）；② 接摄像头跑 G2 正式档；③ 会商项 A/B 定案（决定帧率与语言层演示口径）。

---

## 9. 参赛方案（传智杯 AIoT 嵌入式系统创新大赛）

### 9.1 赛道与组别

2026 第九届传智杯 · AIoT 嵌入式系统创新大赛（id=10036）；**B 组（非 985/211 普通本科）**，可自愿升 A 组/研究生组，严禁降级报错。

### 9.2 五个演示时刻（方案组织核心）

| # | 时刻 | 触发 | 屏上效果 | 底座 |
|---|---|---|---|---|
| ① | **一句话布控**（旗舰） | 看板输入"识别有人在实验室门口逗留超 2 分钟""找出鼓掌的人" | ≤10s 解析卡（闭集/开放/阈值）→ 热部署 → 行为发生 ≤5s 告警卡 + 时间轴打标 + 证据帧跳转 | 文本 LLM 解析 + 三级决策 + crop 演示类兜底 + MQTT `rule/set` 下行 |
| ② | **点谁说谁**（AI 旁白） | 鼠标点跟踪目标 | ≤1s 旁白字幕："该人员近 90 秒三次接近围栏，当前判定徘徊（0.61），crop 复核维持，置信度爬升至 0.83" | 感知事实编译器 → 文本 LLM 合成（**不走图像→VLM**，绕开视觉悬崖） |
| ③ | **和录像聊天** | "刚才 5 分钟门口发生了什么？有人吸烟吗？" | ≤10s 文字回答 + 证据截图卡跳转 | SceneIndex（BM25+bge）+ 叙述日志 + LLM 合成 |
| ④ | **让 AI 说出自己的思考** | 常开字幕带（认知过程视图开关） | 灰区判定实时展开：`主路径 徘徊 0.61 → 灰区 → crop 复核 吸烟=否/徘徊=是 → 翻案 维持↑0.83 → 布控命中 → 告警` | 三级决策 + crop 仲裁 + 否决/采纳逻辑（已实现，缺的只是呈现） |
| ⑤ | **底牌数据** | 常驻角标 | fps / 内存 / NPU 占用 / **板级功耗（npu-smi 直出）** + "单盒·本地·无云·国产芯片"徽章 | 遥测 topic 1s 周期 |

每个时刻都有备用剧本（预置口令/预生成旁白/降级时间线/事件触发），且**走真实代码路径的降级入口，不是假数据**。

### 9.3 算法创新叙事（四支柱）

1. **零标注冷启动（VLM-as-Teacher 蒸馏）**：23,637 伪标签 → 毫秒级边缘模型；主路径 85.7%、+19.1pp；教师同源偏差如实披露；
2. **语言只做语言、视觉保持确定性**：微型文本 LLM + crop 确定性仲裁，正面回应"小模型视觉不可靠"——从失败数据里长出来的架构；
3. **一句话布控**：自然语言 → 边缘规则编译链（闭集映射 + 开放描述 + 演示类扩展）；
4. **可解释复核链可视化**：边缘 AI 决策过程实时可观测，同赛道几乎无竞品。

（4GB 盒：VL-2B stretch 无内存资格，不启用；8GB 到位才评估第五支柱。）

### 9.4 技术性证据表（全部实测）

| 证据 | 数据源 | 关键数字 |
|---|---|---|
| 单盒帧率/延迟 | bench300 盒上复跑 | 1.82fps / p50 546ms（x86 降级 22.6fps/37.5ms） |
| 模型量化对比 | FP32-ONNX vs INT8-OM | mAP 损失/ms/内存三行表 |
| 视频硬解收益 | VDEC vs 软解 CPU | 对比条形图 |
| 端侧语言模型 | tok/s、内存、首解析延迟 | 0.5B：p50 3.9–8.6s、valid 96–100%；1.5B 2.6 tok/s 否决 |
| 功耗/内存 | npu-smi + /proc/meminfo | 空闲/推理/推理+LLM 三档 |
| 视觉判定精度 | 现有评估体系 | 主路径不降；crop per-class P/R；假告警 ≤32/h |
| 隐私/无云 | 部署拓扑图 | 全本地，零公网出口 |

### 9.5 六维作战地图

| 维度 | 权重 | 打法 | 预期 |
|---|---|---|---|
| 技术性 | 35% | 证据表全实测 + 分层认知架构论述 | 优档（25–30） |
| 创新性 | 25% | 四支柱 + 时刻④稀缺性 | 优档 |
| 完整性 | 10% | **接受折损**：软件完整 + 轻壳规范（亚克力/理线/标签，半天） | 可档（5–6） |
| 实用性 | 10% | 真实摄像头/场景补拍、误报率、布控即战力 | 良档 |
| 商业价值 | 10% | BOM：单盒+摄像头 ~¥1700 vs 云年费 TCO 一页纸 | 良档 |
| 社会价值 | 10% | 校园防欺凌 + 养老跌倒（官方点名）+ 隐私不出域 | 良-优 |

### 9.6 交付物（7 必填项）

① 详情及用法（架构图 + 接线 + 看板操作）；② 项目亮点（四支柱 + 实测表）；③ AIoT 概述（摄像头/ MQTT topic 表/ OM+crop+LLM）；④ 关键技术（架构图 + 代码片段 + "开发板+摄像头集成方案"）；⑤ 图片（实物/五时刻/角标）；⑥ 过往获奖（A28、大创 1562 据实填）；⑦ 优化说明（自研全链路 + 本届增量 + 依赖署名）。附件：ZIP ≤200MB；B 站视频 5–8 分钟（开场底牌 15s → 时刻① → ④ → ② → ③ → ⑤）。

### 9.7 参赛风险与评委问答

| 风险/质疑 | 应对 |
|---|---|
| VDEC 不通 | W1 第一验证；退 RTSP 低分辨率软解（1–2fps）或双盒 |
| 文本 LLM 慢/不稳 | G5 早验；0.5B/预生成旁白 |
| crop 新类样本少 | 优先稳定类；开放类如实标"部分支持" |
| "这是不是偏软件" | "单颗国产 8TOPS NPU 完成硬解-推理-语言生成-联网全链路、无服务器无云；受限算力下的分层认知架构是设计核心" |
| "摄像头也算传感器？" | 是——图像传感器，展示采集→网络→VDEC→NPU→MQTT 全链路 |
| "功耗怎么证" | npu-smi 板级实测 + 三档曲线 + 角标常驻 |
| 演示现场翻车 | 每时刻剧本 + 多拍优选 + 彩排三次（W4/W5/W6） |
| 开源合规 | 依赖署名 + 许可证核对（YOLO/Qwen/CANN/Streamlit） |

### 9.8 待确认项（W1 清零）

1. 摄像头方案：复用现有 RTSP IP 摄像头（推荐 H.264 主档匹配 VDEC）/ 新购 ¥100–200 / USB UVC 备用；
2. Atlas 内存：✅ 4GB（11-05 确认），0.5B 常驻，VL-2B 不启用；
3. 作品命名候选：「慧眼智警·边缘认知盒」「哨兵盒 SentinelBox」「问瞳」；
4. 分工四条流：算法上盒（W2/W3）/ 呈现层（W3/W4）/ 测试与数据 / 文档与视频。

---

## 10. 现场清单与演示日

### 10.1 行前桌面推演（10 分钟）

逐条跑 g1/env_probe/g2(testsrc)/export_om(SKIP)/g4/g5/g6/bench，熟悉判定字段；U 盘核对：bundle + sha256 + llama.cpp + GGUF(1.5B/0.5B/2B) + aarch64 wheels + mosquitto deb + 打印清单 + 12V 电源×2 + 摄像头 + 网线。

### 10.2 到场 30 分钟（顺序执行）

```bash
cd /opt/csa/campus_security_agent
sha256sum -c edge_migration/csa_edge_bundle_*.tar.gz.sha256
bash edge_migration/scripts/g1_mem.sh          # variant
bash edge_migration/scripts/env_probe.sh       # atc/pyacl 非 NA
python edge_migration/scripts/g2_decode_bench.py --source rtsp://<子码流> --backend both --frames 300
bash tools/export_om_atlas.sh                  # fails=0 + parity_pass
python edge_migration/scripts/g4_crop_latency_bench.py --model models/crop/crop_v1.onnx --threads 2
python edge_migration/scripts/g6_gru_bench.py --threads 2
python edge_migration/scripts/bench_edge.py --frames 300
# systemd 后：curl -s localhost:8000/api/v1/metrics   → power_w 非 null, npu_mock=false
python edge_migration/scripts/test_edge_engine_smoke.py --frames 120
python edge_migration/scripts/test_language_layer.py
```

### 10.3 三条判读纪律

1. 只看加粗判定字段，不人眼读日志（脚本内置布尔判定）；
2. G2 必须真摄像头 RTSP 子码流，testsrc 数字不算数；
3. JSON 自动落 `results/edge_m0/` + `results/edge_test/`，每天整目录拷 U 盘——不达线的数字也是证据。

### 10.4 必须停下来会商的两条线

- ⛔ G2 三档全超 1.5 核 → 双盒回退评审；
- ⛔ 帧率 stride 最激进仍 <2fps → 演示口径重新定档。

### 10.5 演示日操作清单

```
[ ] 上电：电源→路由器→相机→Atlas（systemd 自动拉起 mosquitto→llm→engine→ui）
[ ] curl :8001/v1/models ｜ curl :8000/health ｜ npu-smi info
[ ] 看板角标四值刷新、认知视图开启、画面叠加正常
[ ] 布控口令三选一预演（不占用演示时段）
[ ] RTSP 拔插一次验证重连（备用剧本已启用则跳过）
[ ] 结果盘 >80% 时清理旧截图
[ ] 复位：POST /api/v1/analyze/stream 新 task（begin_task 全旁路重置）
```

---

## 11. 当前状态与下一步（截至 2026-11-19）

**已完成**：全部代码任务（13 批）、五时刻前端与后端、parity 93.0%、crop 真模型（val 92.4%）、systemd/存储治理/bench 工具、盒上首日实测与 D1–D8 回流。

**未闭环（按阻塞度排序）**：

| # | 事项 | 影响 | 下一步 |
|---|---|---|---|
| 1 | **pyACL 旧式约定 → OM 无法加载** | fps 卡 1.82，所有 NPU 证据缺位 | 跑 `g3_pyacl_probe.py` → shim 修复验证 → export + bench（预期 4+fps） |
| 2 | 真摄像头 RTSP 复测 G2 | 解码结论仅文件档 | 摄像头到位后一条命令 |
| 3 | crop 正式重训（监控域困难负样本 + laying 重标） | parity 残余 48 差异 | 数据工程，x86 GPU |
| 4 | pose OM（QLinear opset11 不支持） | 姿态仍在 CPU | 转 FP 版或 Q/DQ 导出 |
| 5 | sudo 段（mosquitto/llm/engine 服务） | 功耗证据与自启动 | 按 `SERVICE_SETUP_FOR_SUDO.md` |
| 6 | 五时刻连排 ×3 + 评估回归 + 证据表定稿 | 参赛材料 | 依赖 1/2 定档后执行 |
| 7 | 0.5B 正式回归（`llm_json_regression.py` 100 prompts） | 语言层口径 | 盒上跑 |
| 8 | 装配/拍摄/文档 | 交付物 | 按 W6–W8 |

---

**版本**：整合版 v1.0（源自 EDGE_MIGRATION v3.0 / AIoT_COMPETITION v3.0 / edge_migration 系列）
**更新日期**：2026-11-19
