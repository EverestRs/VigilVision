# 模型文件：放置、来源与导出

> 本文件随仓库分发：`models/` 目录只保留这份说明，**权重文件不入库**（从 GitHub Release 下载或按下表官方来源获取）。
> 放置路径简表另见 [05_OPERATIONS.md §2](../docs/05_OPERATIONS.md)。

## 0. 权重从哪里来

**方式一（推荐）：GitHub Release** — 自训权重（PoseClassifier 等）打包在本仓库 Release 附件中，下载后解压/放置到 `models/` 对应路径。

**方式二：官方来源下载或导出**：

| 模型 | 来源 | 许可 |
|---|---|---|
| YOLOv8n/x `.pt` | Ultralytics 官方 <https://github.com/ultralytics/ultralytics> | AGPL-3.0（再分发/商用注意，见 [`01_PRODUCT.md` §7.2](../docs/01_PRODUCT.md)） |
| ViTPose WholeBody | ViTPose 官方仓库 <https://github.com/ViTPose/ViTPose>（model zoo） | 以官方仓库 LICENSE 为准 |
| SlowFast | <https://github.com/facebookresearch/SlowFast>（Kinetics-400 预训练权重） | 以官方仓库 LICENSE 为准 |
| bge-small-zh-v1.5 | BAAI <https://huggingface.co/BAAI/bge-small-zh-v1.5>（ModelScope 有同步） | MIT |
| Qwen2.5-VL-7B-Instruct | ModelScope <https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct> | 以模型页为准 |

**Qwen2.5-VL-7B 不在 Release 中**（约 31GB；权重分片单文件 1.1–3.7GB，超出 GitHub Release 单文件 2GiB、单 Release 合计 10GB 的上限）：请单独下载，放置到**与项目目录同级**的 `Qwen2.5-VL-7B-Instruct/`（即 `configs/vlm_config.yaml` 中 `model_path: ../Qwen2.5-VL-7B-Instruct` 指向的位置）。

**演示/评测视频**（`data/`、`test/`，与项目同级）不在仓库与 Release 中：含真人出镜画面，出于隐私不公开分发，请自备或走私有渠道获取。

## 1. 放置路径

请将真实模型按如下路径放置（CPU ONNX 优先）：

- `models/detector/yolov8x.onnx`
- `models/detector/yolov8n_int8.onnx` (realtime 档位可选)
- `models/pose/vitpose_wholebody_s_int8.onnx` (realtime)
- `models/pose/vitpose_wholebody_b.onnx` (high_accuracy)
- `models/action/slowfast.onnx`
- `models/action/slowfast_int8.onnx` (realtime 档位可选)
- `models/tracker/reid.onnx` (可选)

## 2. 动作模型说明

- `src/core/action_recognizer.py` 已支持基于目标框裁剪的视频片段输入（SlowFast ONNX）。
- 支持单输入或双输入（slow/fast pathway）ONNX；双输入时会按输入名包含 `slow` 自动走慢路径采样。
- 若动作 ONNX 不存在或输入签名不匹配，将自动回退到模拟输出，并在 `metrics.action_debug.error` 中给出原因。
- `models/action/slowfast.pth` 已清理（SlowFast 已禁用）；如需重新导出，请从官方仓库下载 pth 后运行 `tools/export_action_onnx.py --pth <path>`。
- 当前通过 `tools/export_action_onnx.py` 从 SlowFast pth 导出的模型签名为：
	- `slow_pathway: [batch, 3, 8, 224, 224]`
	- `fast_pathway: [batch, 3, 32, 224, 224]`
	- `logits: [batch, 400]`
- 注意：`logits=400` 说明这是 Kinetics 预训练分类头，不是本项目 8 类告警头；直接上线前需做类别映射策略或替换为 8 类微调模型。

## 3. 临时 400→8 映射策略（已实现）

- `src/core/action_recognizer.py` 支持：
	- 先取 top1 外部类别；
	- 若不在项目 8 类内，则按关键词把 Kinetics-400 标签归并到 `H01/H02/H03/H04/H05/M01/S01`；
	- 若仍无法归并，则回退为 `Normal`。
- 400 类标签来源优先级：
	- `action.source_labels_path` 指向的标签文件；
	- 若未提供则尝试使用 `torchvision` 内置 `KINETICS400` 标签。
- 该策略适合快速联调和演示，不等价于 8 类监督训练；比赛提交建议使用 8 类微调权重替换。

## 4. 放置后自检

请执行：

- `python tools/preflight_check.py --profile realtime`
- `python tools/preflight_check.py --profile high_accuracy`

自动导出并重命名检测 ONNX：

- `powershell -ExecutionPolicy Bypass -File tools/export_detector_onnx.ps1`

并访问 `GET /api/v1/preflight` 查看后端预检结果。
