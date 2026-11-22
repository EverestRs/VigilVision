# VigilVision · 慧眼智警

面向校园安防的**开放式视频事件感知与告警平台**：实时完成人员检测、跟踪、姿态估计与行为识别（11 类内置），支持**一句话定义任意新告警规则**，并在本地完成"告警触发 → 证据留存 → 事件裁决 → 历史检索"闭环。

边缘单盒方案（华为 Atlas 200I DK A2 / 昇腾 310B，全本地、≤20W）已完成盒上验证并**归档**：边缘**运行时**代码（`edge_migration/`、`deploy/`、crop 权重、OM/MQTT/语言层接线）已于 2026-11-19 移除；crop **训练工具与数据集保留**（服务器侧 M01 确认钥匙候选）。方案与实测记录见 [docs/04](docs/04_EDGE_AIOT.md)。

---

## 核心指标（真实评估）

| 指标 | 数值 |
|---|---|
| 主路径准确率 | 85.7% / 77.7%（VLM 伪真值）；**94.4% / Macro-F1 0.971**（89 条人工金标） |
| 对比零样本基线 | +19.1pp（entrance_3） |
| 开放事件命中率 | 78%（挥手 100% / 握手 88% / 投掷 75% / 鼓掌 50%，训练未见四类） |
| 跌倒误报抑制 | 89 次 → 6 次（强制 VLM 复核） |
| 吸烟检出 | 0 → 177 帧（1200 帧验证） |
| 主路径推理 | ~10ms/帧（PoseClassifier ONNX，CPU） |
| 全流水线 | 363ms 均值 / 631ms p95 / 2.8fps（bench300） |
| 事件级假告警 | 2.46–31.9 条/小时（分视频/配置） |
| 边缘单盒 | 1.82fps / p50 546ms（4GB Atlas，OM 待修复） |

> 伪真值与主路径同源，指标存在乐观偏差；人工金标与事件级评估为独立锚点。口径详见 [docs/03](docs/03_QUALITY_AND_ROADMAP.md)。

---

## 技术主线

1. **VLM-as-Teacher 弱监督**：本地 Qwen2.5-VL 为未标注视频生成 23,637 条伪标签 → 训练轻量姿态序列分类器（PoseClassifier），零人工标注，CPU ~10ms；
2. **分层开放行为识别**：PoseClassifier 主路径（≥0.75 采纳）+ VLM 灰区复核（0.40–0.75）+ SlowFast 回退（<0.40）+ 强制复核（H03/M01）；体系外行为用一句话规则 + 开放词汇判定；
3. **全本地部署**：检测/姿态/行为 ONNX CPU 推理，大模型走本地 vLLM/llama-server，视频不出域。

---

## 架构一图

```
视频源(本地/RTSP)
  → Detector(YOLOv8n) → Tracker(ByteTrack) → PoseEstimator(ViTPose 133 点)
  → ActionRecognizer 双路径
       主：PoseClassifier(32 帧姿态 → 11 类)
       备：SlowFast + Kinetics-400 关键词映射
  → 三级决策 + 强制 VLM 复核(H03/M01)
  → BehaviorReasoner(S01/S02/M01) → InteractionAnalyzer → TemporalAdjudicator(默认关)
  → NLAlertRuleEngine(开放词汇) → AlertSystem → AlertAdjudicator(裁决)
  → SceneNarrator(叙述) → SceneIndex(BM25+bge 检索)
  → FastAPI + WebSocket + Streamlit
```

细节见 [docs/02](docs/02_ARCHITECTURE.md)。

---

## 文档导航

| 文档 | 内容 | 适合谁 |
|---|---|---|
| [docs/01_PRODUCT.md](docs/01_PRODUCT.md) | 背景、行为类别体系、功能清单、指标、档位、数据策略、VLM-as-Teacher、安全、交付物 | 产品/评审 |
| [docs/02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md) | 总体架构、模块设计、推理流水线、配置、数据模型、API、算法细节、工程细节 | 工程师 |
| [docs/03_QUALITY_AND_ROADMAP.md](docs/03_QUALITY_AND_ROADMAP.md) | 评估体系与结果、缺陷修复总账、已知局限、未见行为路线图、下一步优先级 | 技术负责人 |
| [docs/04_EDGE_AIOT.md](docs/04_EDGE_AIOT.md) | 边缘 v3.0 方案、Atlas 硬件与 M0、模块改造、里程碑、13 批进展、盒上实测、参赛五时刻（**方案存档，代码已移除**） | 方案追溯 |
| [docs/05_OPERATIONS.md](docs/05_OPERATIONS.md) | 环境、模型、启动、训练、评估、工具索引、systemd、排障表 | 运维/演示 |
| [docs/06_GAPS_DESIGN.md](docs/06_GAPS_DESIGN.md) | 缺口设计：视频切片（双路径 + 编码约定 + 验收）、render H.264 修复、多路/安全待展开项 | 设计/实现 |
| [models/README.md](models/README.md) | 模型放置路径、下载来源（Release/官方）、ONNX 签名、自检命令 | 开发/部署 |

---

## 目录结构

```
campus_security_agent/
├── src/
│   ├── core/          # detector/tracker/pose_estimator/action_recognizer/
│   │                  # behavior_reasoner/interaction_analyzer/temporal_adjudicator/
│   │                  # vlm_reviewer/nl_rule_engine/scene_narrator/scene_index/
│   │                  # alert_adjudicator/inference_backend/crop_classifier/
│   │                  # fact_compiler/track_narrator/decision_trace
│   ├── pipeline/      # video_stream/frame_buffer/inference_engine/alert_system/
│   │                  # vlm_router/mqtt_bridge
│   ├── api/           # rest_api / websocket_server
│   ├── evaluation/    # evaluator
│   ├── training/      # pose_classifier
│   └── utils/         # config_loader / preflight / schemas / npu_telemetry
├── configs/           # model_config / behavior_rules / vlm_config / camera_config / labels
├── models/            # 模型权重（需自行下载/导出，见 docs/05 §2）
├── demo/web_interface # Streamlit 看板
├── tools/             # 训练/评估/导出/回归/冒烟脚本
├── scripts/           # start_vllm.sh 等
├── results/           # 事件/截图/叙述日志/评估报告
├── data_external/     # HMDB51 未知事件测试集等
├── docs/              # 整合后的 5 篇文档 + archive
├── run.py / train.py / eval.py
└── requirements.txt / environment.yml
```

---

## 快速开始

```bash
# 1. 环境
conda env create -f environment.yml && conda activate VdAnalyst

# 2. 放置模型：自训权重从本仓库 GitHub Release 下载，第三方权重来源见 models/README.md §0
#    （演示/评测视频不在仓库内，需自备到 data/ 与 test/，说明见 models/README.md）
python tools/preflight_check.py --profile realtime

# 3. 启动大模型（可选，VLM 复核/叙述/规则需要）
nohup bash scripts/start_vllm.sh > results/vllm_serve.log 2>&1 &

# 4. 启动 API + 前端
INFERENCE_PROFILE=realtime python run.py
streamlit run demo/web_interface/app.py

# 5. 跑一段视频
curl -X POST localhost:8000/api/v1/analyze/stream \
  -H 'Content-Type: application/json' \
  -d '{"stream_url":"data/demo_video.mp4","duration":60}'
```

> 边缘单盒（Atlas 200I DK A2）的部署与探针命令仅保留在 [docs/04](docs/04_EDGE_AIOT.md) 作为方案记录，仓库内已无对应代码。

---

## 当前状态（2026-11-19）

| 线 | 状态 |
|---|---|
| 服务器主路径 | v2 PoseClassifier 已部署（训练/推理同口径，best val 69.88%）；双口径 94.4% |
| 场景/开放词汇/裁决 | 全部上线；11 项代码审计问题已修复；golden 回归护栏就位 |
| 告警证据切片 | 本地文件源已实现（前 5s + 后 5s，H.264，事件回写）；RTSP 流源待接旁路录制 |
| 未见行为 | Stage 1/2 验收通过；Stage 3（闭集自扩展）样本不足未达标，生产保持 11 类 |
| 边缘盒 | 已完成盒上验证并**归档**：13 批代码任务、parity 93.0%、首日实测 1.82fps；代码已移除，记录见 docs/04 |
| 参赛 | 省赛截止 2026-11-18 已到期；材料/拍摄/证据表待收尾（见 docs/04 §11） |

---

## 已知短板（如实声明）

- 11 类中仅 5 类有有效训练覆盖；H02/M03/S02 主路径结构性无法输出；M02 教师标签虚多（P=0）；
- 远距小目标（bbox 75–150px）VLM 判定 50/50 抖动，属 7B 感知极限；
- 单路处理、无 API 认证、无视频切片导出、Docker 不做；
- 帧级误报率 76–81%，事件口径假告警 2.46–31.9 条/小时。

---

## 许可与依赖注意

YOLOv8（Ultralytics）为 **AGPL-3.0**，商用/闭源分发需评估；ViTPose / SlowFast / ByteTrack / ONNX Runtime / FastAPI 为 MIT，Streamlit 为 Apache-2.0，Qwen2.5-VL 按官方模型许可。数据集：Kinetics-400、COCO、HMDB51（研究用途）。
