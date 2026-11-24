# 05 · 运行手册

> 合并自：`PRD.md` §10、`TECHNICAL_REPORT.md` §10、`models/README.md`、`tools/README.md`、`deploy/README.md`、`edge_migration/scripts/ENV_LOCK.md`，以及各文档中的排障记录
> 最后更新：2026-11-19 ｜ 适用：x86 服务器（开发/训练/评估）与 Atlas 200I DK A2 单盒（演示）
> 相关文档：[02 架构](02_ARCHITECTURE.md) ｜ [04 边缘化](04_EDGE_AIOT.md)

---

## 1. 环境准备

### 1.1 x86 服务器（开发/训练/评估）

| 项 | 要求 |
|---|---|
| OS | Linux（Ubuntu 20.04+）或 Windows |
| Python | 3.10 |
| 推理硬件 | CPU（x86_64，建议 6 核+）；训练需 V100 32G ×1 |
| 内存 / 存储 | ≥ 8GB / ≥ 5GB（含模型） |
| 大模型 | 本地 Qwen2.5-VL-7B + vLLM（需 GPU） |

```bash
conda env create -f environment.yml
conda activate VdAnalyst
# 环境内含 numba/pandas/hdf5plugin（若报缺）；bitsandbytes>=0.45（bnb 量化可选）
```

### 1.2 Atlas 盒（aarch64，方案存档）

> 边缘代码已于 2026-11-19 移除，本节命令仅供重装/追溯参考，见 [04](04_EDGE_AIOT.md)。

```bash
# 盒上环境（实测路径 ~/.conda/envs/csa）
conda create -n csa python=3.10 -y
pip install -r requirements-aarch64.txt
# D5 补齐（引擎 import 链实际需要，原清单漏项）：
pip install torch torchvision transformers jieba rank_bm25 onnx
# 系统包：mosquitto（7 个 deb）、llama.cpp（源码编译）
```

**版本纪律**：CANN / 固件 / pyACL / llama.cpp 四件套版本一次锁定并记录（原 `ENV_LOCK.md` 随边缘代码归档，见 [04](04_EDGE_AIOT.md)）；升级须重跑探针与冒烟。

---

## 2. 模型准备

### 2.1 放置路径

| 模型 | 路径 | 用途 |
|---|---|---|
| YOLOv8n | `models/detector/yolov8n.onnx`（FP32）/ `yolov8n_int8.onnx`（realtime）/ `.om`（边缘） | 检测 |
| ViTPose WholeBody | `models/pose/vitpose_wholebody_b.onnx`（高精度）/ `vitpose_wholebody_s_int8.onnx`（实时） | 姿态 133 点 |
| SlowFast | `models/action/slowfast.onnx` / `slowfast_int8.onnx` | 备用行为路径 |
| PoseClassifier | `models/action_v2/pose_classifier.onnx` + `pose_classifier_best.pth` | **主路径（生产）** |
| ~~Crop 分类器~~ | ~~`models/crop/crop_v1.onnx`~~ | 边缘仲裁（已归档移除） |
| bge-small-zh | `models/embedding/bge-small-zh-v1.5` | 检索/聚类（缺失降级 BM25） |
| ReID（可选） | `models/tracker/reid.onnx` | 未启用 |

### 2.2 获取与导出

> 各模型的下载来源（官方链接 / 本仓库 Release）见 [models/README.md §0](../models/README.md)：自训权重（PoseClassifier 等）从本仓库 **GitHub Release** 附件下载；第三方权重按官方来源获取。演示/评测视频（`data/`、`test/`）不在仓库内，需自备（含真人画面，不公开分发）。

- **YOLOv8n**：Ultralytics 官方 `.pt` → `tools/export_detector_onnx.ps1`（Windows）或 ONNX 导出脚本；INT8 量化另行导出；
- **ViTPose**：官方权重 → ONNX 导出（`tools/export_pose_int8_onnx.py` 量化）；
- **SlowFast**：`models/action/slowfast.pth` 已清理（运行时只用 ONNX）；如需重导出，从官方仓库下载 pth 后执行 `tools/export_action_onnx.py --pth <path>`（签名：`slow_pathway [b,3,8,224,224]`、`fast_pathway [b,3,32,224,224]`、`logits [b,400]`）；
- **PoseClassifier**：`train.py` 训练后自动导出（产物含 `train_metrics.json`）；
- ~~Crop / 边缘 OM 导出~~：`train_crop_classifier.py`、`export_om_atlas.sh` 已随边缘代码移除。

> SlowFast 输出的 400 类是 Kinetics 预训练头，通过关键词映射归并为 11 类（仅作回退）；正式精度以 PoseClassifier 为准。

### 2.3 预检

```bash
python tools/preflight_check.py --profile realtime
python tools/preflight_check.py --profile high_accuracy
# 或 REST：GET /api/v1/preflight
```

预检自动探测 ONNX 输入输出签名；模型缺失不阻塞启动（优雅降级为模拟/空结果，`is_real_model_loaded=false`）。

---

## 3. 启动与访问

### 3.1 三服务（服务器）

```bash
# 1) 大模型服务（后台，日志 results/vllm_serve.log）
nohup bash scripts/start_vllm.sh > results/vllm_serve.log 2>&1 &
#    V100 需 --dtype half；--limit-mm-per-prompt image=8；gpu_mem_util=0.60 预留 ORT 显存
#    可选 bnb 低占用：VLLM_QUANT=bnb VLLM_GPU_MEM_UTIL=0.30 bash scripts/start_vllm.sh

# 2) API + 引擎
INFERENCE_PROFILE=realtime python run.py          # 默认 :8000

# 3) 前端
streamlit run demo/web_interface/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

### 3.2 远程访问

```bash
ssh -N -L 8501:localhost:8501 -L 8000:localhost:8000 用户@服务器
# 浏览器打开 http://localhost:8501
```

### 3.3 边缘档（已归档）

> `INFERENCE_PROFILE=edge`、盒上 LLM 启动脚本与 `/api/v1/metrics` 遥测端点已随边缘代码移除；重建路径见 [04](04_EDGE_AIOT.md)。

健康检查：

```bash
curl -s localhost:8001/v1/models | head -c 120   # LLM 在线
curl -s localhost:8000/health                    # 引擎在线
```

---

## 4. 训练

### 4.1 PoseClassifier（主路径）

```bash
# 标准训练（数据来自 VLM 伪标签 manifest）
python train.py --epochs 100 --batch-size 64
# 产物：models/action_v2/pose_classifier_best.pth + pose_classifier.onnx + train_metrics.json

# v2 复现（当前生产口径）
python tools/train_v2.py --extra results/training_data_v2/h03_extra/class_H03_extra.npz --device cuda
```

> **训练数据布局（2026-11-19 迁移）**
> - `results/training_data_v2/`：当前口径（1,235 样本；npz 已平铺、manifest 字段为 int/float；含 `h03_extra` 2,485 条与 `prototype_classes` 4 类）；
> - `results/training_data/`：**仅保留 v1 全量 manifest**（23,637 条，供 `relabel_h03` / `eval_dual_caliber` 做标签统计，不参与训练），旧 npz（约 190M）已归档至项目外备份；
> - 默认入口已切 v2：`train.py`、`tools/finetune_with_extra.py`、`tools/extend_pose_classifier.py`；输出目录 `build_prototype_training_data.py` / `dump_pose_sequences.py` 同步切到 v2；
> - 迁移脚本 `python tools/migrate_v2_layout.py`（幂等，可复现布局转换与字段类型规范化）。

关键注意：
- 训练/推理必须同口径（bbox 中心 + 对角线归一化；conf clip 0–1）；
- 类别不均衡用加权 CrossEntropy（逆频加权 clip 0.5–10）或 focal loss；
- 按视频分层 15% 验证，避免同视频泄漏。

### 4.2 Crop 分类器（服务器侧确认钥匙候选）

训练/导出工具**保留**（无边缘依赖）：`tools/train_crop_classifier.py`、`tools/tune_crop_thresholds.py`，数据集 `results/crop_dataset/`（7 类，Normal/M01/M02/H03/Wave/Throw/Handshake）。

```bash
# 构建数据集（数据源：results/vlm_labels_v2 + results/vlm_smoking_relabel + HMDB51）
python tools/train_crop_classifier.py build --labels "results/vlm_labels_v2/*_vlm_labels.json" --out results/crop_dataset
# 训练 / 导出
python tools/train_crop_classifier.py train  --data results/crop_dataset --epochs 30
python tools/train_crop_classifier.py export --ckpt results/crop_dataset/best.pth --out models/crop/crop_v1.onnx
# 阈值扫描（输出各满足目标精度的最低阈值）
python tools/tune_crop_thresholds.py --ckpt results/crop_dataset/best.pth --data results/crop_dataset --target-precision 0.80
```

**启用前需补齐两件**（边缘运行时已移除）：
1. **模型权重不恢复**：`models/crop/crop_v1.onnx` 是边缘域训练产物（val_acc 92.4%，val 每类仅 9 张），服务器监控域应重训；
2. **推理封装需自建**：原 `edge_migration/crop_classifier.py`（letterbox 224 + 缓存 + 阈值仲裁）已删；预处理规则可参考 `train_crop_classifier.py` 的 `_letterbox`（pad=114 → BGR2RGB → /255），阈值候选见 `results/crop_dataset/thresholds_tuned.json`。`vlm_config.crop_review` 段已删，启用时需重新加回。

### 4.3 闭集扩展（Stage 3，未达标）

```bash
python tools/dump_pose_sequences.py --new-data ...      # 真实片段提取滑窗
python tools/extend_pose_classifier.py --new-data ... --device cuda --deploy
# 门槛：每新类 ≥30 条滑窗；未达标生产保持 11 类（回滚备份 .bak11）
```

### 4.4 数据重标与训练数据重生成

```bash
python tools/relabel_smoking.py      # M01 重标（改善 prompt + person crop）
python tools/relabel_h03.py          # H03 候选重标
python tools/regen_training_data.py  # 按当前管线口径重生成 NPZ
python tools/extract_action_training_data.py
```

---

## 5. 评估

### 5.1 伪真值生成 + 三组评估

```bash
# 1) 教师标注（conf ≥0.6 过滤）
python tools/vlm_label_pipeline.py --data-dir ../test --output-dir results/vlm_labels_test \
    --vllm-url http://localhost:8001/v1/chat/completions --workers 6
python tools/build_eval_ground_truth.py --labels-dir results/vlm_labels_test --output-dir results/eval/gt

# 2) 主路径 / 基线 / 上下文消融
python eval.py --video ../test/whole_process_corner_3.mp4 --ground-truth results/eval/gt/whole_process_corner_3.json
python eval.py --video ... --ground-truth ... --baseline
python eval.py --video ... --ground-truth ... --no-scene-context

# REST 后台：POST /api/v1/evaluate → GET /api/v1/evaluate/{eval_id}
```

### 5.2 双口径（金标校准）

```bash
python tools/eval_dual_caliber.py --gt results/frame_gt/whole_process_corner_1/gt.json ...
# 产物：results/eval/dual_caliber_report.json
```

### 5.3 事件级（必须走部署路径）

```bash
python tools/eval_event_level.py --mode live --video ... --gt results/eval/event_level/...
# 指标：事件召回（检测 ≥3 帧 / 告警 ≥1 次，±3s）+ 假告警/小时
# 产物：results/eval/event_level/*
```

### 5.4 开放事件与叙述

```bash
python tools/eval_open_events.py        # HMDB51 四类，产物 results/eval/open_events/*.json
python tools/eval_narration_consistency.py   # 叙述-检测一致率
```

### 5.5 回归护栏（改决策逻辑必跑）

```bash
python tools/test_golden_regression.py capture   # 采集基线
python tools/test_golden_regression.py check     # 比对（conf ±0.02 / bbox ±2px）
```

---

## 6. 工具脚本索引

| 类别 | 脚本 | 用途 |
|---|---|---|
| 预检/导出 | `tools/preflight_check.py` | 模型与档位预检 |
| | `tools/export_detector_onnx.ps1` | YOLO → ONNX |
| | `tools/export_action_onnx.py` | SlowFast → ONNX |
| | `tools/export_detector_int8_onnx.py` | YOLOv8n INT8（realtime 档在用） |
| | `tools/export_pose_int8_onnx.py` | ViTPose INT8 |
| 标注/数据 | `tools/vlm_label_pipeline.py` | VLM 伪标签 |
| | `tools/build_eval_ground_truth.py` | 评估 GT |
| | `tools/gen_frame_gt_pack.py` | 人工金标包（pack/apply） |
| | `tools/gen_event_candidates.py` | 事件候选/终审页/落盘 |
| | `tools/annotation_tool.py` | 标注工具 |
| | `tools/regen_training_data.py` | 按当前管线重生成训练数据 |
| | `tools/inspect_high_value_samples.py` | H01/H02/H04 高价值样本筛选 |
| | `tools/migrate_v2_layout.py` | v2 训练数据布局规范化（一次性，幂等） |
| | `tools/extract_action_training_data.py` | 姿态序列抽取 |
| 训练 | `train.py` / `tools/train_v2.py` | PoseClassifier |
| | `tools/train_crop_classifier.py` | Crop 多分类器（服务器侧确认钥匙候选，模型需重训） |
| | `tools/tune_crop_thresholds.py` | Crop 各类阈值扫描 |
| | `tools/extend_pose_classifier.py` | 闭集扩展 |
| 评估 | `eval.py` | 伪真值三组评估 |
| | `tools/eval_dual_caliber.py` | 双口径 |
| | `tools/eval_event_level.py` | 事件级 |
| | `tools/eval_open_events.py` | 开放事件 |
| | `tools/eval_narration_consistency.py` | 叙述一致率 |
| | `tools/compare_mapping_reports.py` | 映射报告对比 |
| | `tools/summarize_eval_reports.py` | eval_report_*.json 汇总为对比表 |
| 回归/测试 | `tools/test_golden_regression.py` | golden 护栏 |
| | `tools/test_b6_pipeline.py` | 三阶段流水线一致性/吞吐 |
| | `tools/test_tracker_ab.py` | 跟踪 A/B |
| | `tools/smoke_test_scene_modules.py` | 场景三模块冒烟（需 vLLM） |
| | `tools/smoke_test_nl_rules.py` | 一句话布控端到端冒烟（需 vLLM） |
| | `tools/test_clip_export.py` | 切片端到端（文件源 / 流源失败 / 队列满） |
| 性能 | `tools/benchmark_*.py` | 帧率/延迟 |
| | `tools/measure_pipeline_stages.py` | 各阶段耗时 |
| 运维 | `scripts/start_vllm.sh` | LLM 服务 |
| | `tools/diagnose_stream_stats.py` | WS 推送/丢帧统计诊断 |

---

## 7. 盒上部署（systemd + MQTT + 存储，方案存档）

> `deploy/` systemd 单元与 `edge_migration/` 代码已于 2026-11-19 移除；本节保留为盒上重装参考。

### 7.1 systemd（Ubuntu 22.04）

假定项目在 `/opt/csa/campus_security_agent`，Python 用 `/opt/csa/venv/bin/python`。

```bash
sudo cp deploy/csa-vlm.service deploy/csa-engine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mosquitto            # 系统包自带 unit
sudo systemctl enable --now csa-vlm.service      # llama-server（:8001）
sudo systemctl enable --now csa-engine.service   # FastAPI + 引擎（:8000）
# 演示机可选：sudo systemctl enable --now csa-ui.service   # Streamlit :8501
```

纪律：
- 启动顺序 `mosquitto → csa-vlm → csa-engine`（After/Wants）；
- 全部 `Restart=always` + `RestartSec=5`（7×24 自愈）；
- `csa-engine` 的 `ExecStartPre` 做 LLM 健康检查，失败仍启动（引擎自带降级）；
- 环境变量在 unit 内 `Environment=`（`INFERENCE_PROFILE=realtime`；原 edge 档已归档），改配置 = 改 unit + daemon-reload；
- 日志走 journald：`journalctl -u csa-engine -f`。

### 7.2 MQTT（已归档）

> MQTT 桥接（`mqtt_bridge.py`、`model_config.mqtt` 段）已随边缘代码移除；topic 设计见 [04 §4.3](04_EDGE_AIOT.md)。

### 7.3 存储治理（已归档）

> `storage_gc.py` 已移除；当前由 `results/` 人工轮转（演示前检查结果盘水位 <80%）。

---

## 8. 现场操作（摘要，方案存档）

> 以下流程用于边缘盒重装/复测，脚本已移除；完整打印版见 [04 §10](04_EDGE_AIOT.md)。

1. 行前桌面推演：g1 / env_probe / g2(testsrc) / g4 / g5 / g6 / bench，熟悉判定字段；
2. 到场顺序：校验包 → G1 内存 → env_probe → G2 真 RTSP → G3 OM → G4 crop → G5 LLM → G6 GRU → bench → metrics → 两个 smoke；
3. 三条纪律：只看加粗判定字段；G2 必须真摄像头子码流；JSON 每天拷 U 盘；
4. 两条会商线：G2 三档全超 1.5 核；帧率最激进仍 <2fps；
5. 演示日：上电顺序 → 三项 curl/npu-smi → 看板角标/认知视图 → 布控口令预演 → RTSP 拔插 → 清理结果盘 → 新 task 复位。

---

## 9. 故障排查

| 症状 | 根因 | 处理 |
|---|---|---|
| 所有行为都输出 Normal | 姿态缓冲未写入（主路径）/ 17 点污染 / 训练推理口径不一致 | 检查 `append_pose()` 调用；空姿态跳过；确认 v2 模型 + 当前归一化 |
| 姿态关键点乱飞/精度差 | ViTPose 缺 mean/std 或 crop 拉伸 | 确认 D2 修复（ImageNet 归一化 + 1.25 外扩保比例） |
| 单张 `infer` 关键点整片错位（~288px） | `_preprocess` 把归一化结果写回 **uint8** 的 `_rgb_buffer`：负数回绕、正数截断 | 2026-11-19 改为写入 float32 `_tensor_buffer`；修复后 `infer` 与 `infer_batch` L2=0 |
| 批量姿态加速不明显 | 引擎已调 `infer_batch`（一次 `session.run`），耗时大头在 cv2 预处理与逐样本解码，ORT batch 收益有限 | 实测 4 track：44.4ms vs 逐个 54.8ms（1.23x）；进一步提速需流水化（1 帧延迟） |
| `np.stack` 异常被吞 | 缓冲混入 17 点 fallback | 形状校验；fallback 补齐 133 点 |
| VLM 复核结论明显错/编造 | 完整示例 JSON 被照抄；max_tokens 截断；关键词兜底 | 填空式 prompt；`review_max_tokens≥256`；收紧兜底词表 |
| 多规则只回第一条 | 7B 对 JSON 数组偷懒 | 每规则独立调用 |
| vLLM 图像编码静默失败 | 双实例瓜分显存 | `start_vllm.sh` 防重复启动；`--dtype half`（V100） |
| 首帧延迟 5–10× | 未 warmup | 引擎启动时 dummy 推理（B2） |
| 帧丢率高 / p95 高 | VLM 同步阻塞 | 确认异步复核 + 并发槽位 + 缓存（B1/B3/B5） |
| 画面只有 ~2fps、框更新卡顿 | `max_fps` 限速曾加在**每个解码帧**上，与 `frame_stride` 相乘（8/4=2） | 2026-11-19 已改为只对推理帧限速；引擎单帧 ~80ms（12.6fps），瓶颈是节流不是模型 |
| 帧率仍不够（realtime 档） | `frame_stride=4` 把候选帧压到 6.25fps；`ws_sleep_ms=24` 让推送协程 10s 内空睡 4.7s | 已定档 `frame_stride=2 / max_fps=12 / ws_sleep_ms=2`，10s 1080p 实测 7.6–8.1fps、推送延迟 <0.1ms/帧。A/B/C 数据：4/8/24 → 5.3fps；2/12/24 → 9.3fps 但推送睡 4.7s、丢 26 帧；2/12/2 → 7.8fps、丢 17 帧、推送睡 0.4s（延迟优先） |
| 前端框抖动/推送量大 | 每处理帧全量推送，静态场景内容重复 | 已启用差分推送（`ws_diff_push`）：bbox 2px/置信度 0.05 量化签名 + 0.5s 墙钟心跳 + 告警必发。实测静态跳过 81%、微动 63%，CPU 明显下降 |
| 叙述不提已检出的吸烟 | 采样稀疏 + 全图分辨率不足 | 检测事实注入 + `max_image_px 768` + 缓冲 `target_width 960` |
| 告警没有截图 | 历史 bug：NameError + 假路径 | 确认 2026-09-28 修复版本；`snapshot_url` 指向 media 端点 |
| 告警永远 pending | 裁决器单槽/队列满 | 查 `metrics.alert_adjudicate_dropped`；调大 `max_pending_alerts` |
| 检索 `doc_type` 返回空 | 过滤在 top_k 之后 | 确认 #5 修复（先过滤再截断） |
| 检索接口周期性卡顿 | 全量重嵌入 | 确认增量嵌入（md5 指纹） |
| 跨任务前半段无叙述 | 帧号残留 | `begin_task()` 重置四个旁路模块 |
| 环形缓冲长期空 | 存储步长与处理步长互质 | 每处理帧入队 |
| `prompt.format()` KeyError | JSON 示例花括号未双写 | `{{ }}` 转义或 replace 注入；建议 `safe_format` |
| Streamlit 前端组件白屏 | JS 语法/递归错误 | 看红色错误框；改 JS 后跑渲染 + esprima |
| （边缘，已归档）盒上 OM 加载失败 | pyACL 新旧约定不匹配 | 见 docs/04 §8.3；代码已移除 |
| 盒上帧率 <2fps | OM 未跑通；`frame_stride` 不是检测杠杆 | 修 OM；调 `detector_stride` / `action_stride` |
| 盒上 `atc` 不可用 | TBE 依赖不在系统 python | 装到 `/usr/bin/python3.10` user-site |
| 裁剪 `.om.om` 双后缀 | export 脚本 `--output` 带后缀 | 已修，输出不带后缀 |
| `--input-size 160` 无效 | 静态输入真模型 | 改用 `throttle_frames`，或导出动态输入 |

---

## 10. 版本锁定（ENV_LOCK 示例）

`env_probe.sh` 生成，2026-10-01 快照（盒上实测机为 CANN 6.2.T2.0.B133）：

| 组件 | 版本 |
|---|---|
| python | 3.10.13 |
| onnxruntime | 1.23.2 |
| numpy | 1.26.4 |
| opencv | 4.11.0 |
| CANN / atc / pyACL / npu-smi / llama-server | 盒上实测时补齐（x86 为 NA） |

---

**版本**：整合版 v1.0
**更新日期**：2026-11-19
