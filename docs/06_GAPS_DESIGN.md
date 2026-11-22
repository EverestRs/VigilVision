# 06 · 缺口设计与实现方案

> 状态：**切片路径 A 已实现并实测**（2026-11-19，`src/pipeline/clip_exporter.py`）；render 编码修复已落地；多路并发 / API 安全为待展开项
> 日期：2026-11-19 ｜ 关联：[02 架构](02_ARCHITECTURE.md) ｜ [05 运行手册](05_OPERATIONS.md)
> 原则：设计必须能落到文件/函数/命令，验收必须可跑；不做"只写在文档里的架构"。

---

## 1. 视频切片导出（完整设计）

### 1.1 现状与缺口

| 项 | 现状 | 位置 |
|---|---|---|
| 数据字段 | `Alert.clip_path: str = ""`，告警与规则事件均写空串 | `src/utils/schemas.py:33`、`alert_system.py:66`、`nl_rule_engine.py:332` |
| 触发帧 | `build_alert(..., frame=frame)` 已持有告警触发帧，仅用于截图 | `src/pipeline/alert_system.py:49-62` |
| 缓冲 | `frame_buffer`（降采样 480/960，30s）+ `frame_buffer_orig`（原分辨率，12s / 48 帧 / 2 samples/s） | `src/pipeline/inference_engine.py:160,205-210` |
| 写盘 | ✅ 已实现：`src/pipeline/clip_exporter.py`（路径 A：文件源切片 + 事件回写）；`rest_api.py` 的 `cv2.VideoWriter` 属于 **render 任务**，不是切片 | 本次 |

**缺口已补齐**：字段、触发帧、缓冲 + 窗口计算/编码封装/异步落盘/事件回写全部就位（路径 A）；路径 B 的流源旁路录制仍未接线（见 §1.6）。

### 1.2 三个约束（决定不能用单一工具）

**约束 A：切片来源不同**

| 源类型 | 可行做法 | 不可行做法 |
|---|---|---|
| 本地文件（`stream_url` 是路径） | ffmpeg 无损 remux（`-c copy`），毫秒级、零画质损失 | 全解码 + 重编码（cv2 路径，慢 10–100 倍） |
| RTSP / 摄像头流 | 只能重编码（无存档文件） | remux（没有源文件） |

**约束 B：现有缓冲不是切片源**

`frame_buffer_orig` 是 **2 samples/s、48 帧、12 秒**的处理帧，不是源 25fps 全帧。直接把它写成视频，得到的是 2fps 幻灯片。因此：

- 文件源 → 从原文件按时间戳 remux；
- 流源 → 读取线程旁路录制原始帧（独立于推理节拍），不能用推理缓冲代替。

**约束 C：画框/脱敏会强制重编码**

告警证据若要叠加检测框或人脸模糊，帧已被修改，`-c copy` 不可用，必须重编码。此时编码交给 ffmpeg（rawvideo stdin → libx264），而不是 `cv2.VideoWriter`（mp4v 兼容性差）。

### 1.3 工具分工原则

```
逐帧算法（画框 / 骨骼 / 人脸脱敏）  →  numpy + cv2
编码 / 封装 / 剪切 / 拼接 / 转码     →  ffmpeg
```

不是二选一：cv2 处理像素，ffmpeg 处理码流。

### 1.4 设计：双路径切片

```
告警触发（risk=HIGH 或 NL 规则命中，含 frame_id / video_time）
│
├─ 路径 A：本地文件源 + 无需画框/脱敏
│    ffmpeg -ss <t-5> -t <10> -c copy -avoid_negative_ts make_zero out.mp4
│    ├─ -ss 放 -i 前：快速定位（切点可能提前 ≤1 GOP）
│    ├─ 切点精度不足时（证据需要精确帧）：退 -c:v libx264 -preset veryfast
│    └─ 文件不存在/不可读 → 路径 B
│
├─ 路径 B：流源，或需要画框/脱敏/合成
│    录制线程（源帧率）→ ffmpeg rawvideo 管道
│    -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i pipe:0
│    -an -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart
│
└─ 两者统一：异步任务队列 + 落盘 + 回写 clip_path + 存储治理
```

**窗口语义**：告警前 5s、告警后 5s（可配）。路径 B 需要"未来帧"，因此录制线程必须常开（或告警后延迟 5s 收尾），不能在触发瞬间才启动。

### 1.5 接口设计

新增 `src/pipeline/clip_exporter.py`：

```python
@dataclass
class ClipRequest:
    task_id: str
    frame_id: int
    event_filename: str        # results/event_{fid}.json
    source_url: str            # 本地路径或 rtsp://
    video_time_sec: float      # 触发时刻（视频时间）
    need_overlay: bool = False # 画框/脱敏 → 强制重编码
    before_sec: float = 5.0
    after_sec: float = 5.0

class ClipExporter:
    """异步切片：单工作线程 + 有界队列；失败只记录，不影响主链路。"""
    def __init__(self, output_dir: str, cfg: dict): ...
    def submit(self, req: ClipRequest) -> bool: ...   # 队列满返回 False 并计数
    def status(self) -> dict: ...

def export_clip_file(src: Path, t0: float, duration: float, out: Path,
                     precise: bool = False) -> tuple[bool, str]:
    """路径 A：remux（precise=True 时重编码对齐）。"""

def encode_frames(frames: Iterable[np.ndarray], fps: float, size: tuple[int, int],
                  out: Path) -> tuple[bool, str]:
    """路径 B：rawvideo → libx264/yuv420p/faststart。"""
```

接线点：
- `InferenceEngine._stage3_recognize`：告警 `triggered` 或规则命中时，组装 `ClipRequest` 提交（**不阻塞**）；
- `AlertSystem.build_alert`：保持只做截图；`clip_path` 初始为空，切片完成后由 exporter 回写事件 JSON；
- `FrameResult.alert.clip_path`：异步完成后下次推送自然带出（与 `final_decision` 同语义）。

配置（新增 `configs/clip_config.yaml` 或并入 `model_config.yaml`）：

```yaml
clip_export:
  enabled: false          # 默认关，演示/取证时开
  before_seconds: 5
  after_seconds: 5
  need_overlay: false     # true → 强制重编码（画框/脱敏）
  crf: 23
  preset: veryfast
  max_queue: 8
  keep_days: 7            # 存储治理
```

### 1.6 编码与兼容性约定

| 项 | 约定 | 理由 |
|---|---|---|
| 视频编码 | H.264（libx264） | 浏览器/播放器通用 |
| 像素格式 | `yuv420p` | Safari/Chrome 硬性要求 |
| 封装 | MP4 + `+faststart` | moov 前置，支持边下边播 |
| 音频 | `-an`（默认不带） | 安防证据一般不需要；需要时 `-c:a aac` |
| 帧率 | 路径 A 保持源帧率；路径 B 用录制帧率 | 避免"幻灯片" |
| 码率控制 | `-crf 23 -preset veryfast` | 质量/速度平衡；10s 1080p 约 1–3s |

### 1.7 边界与失败回退

| 场景 | 处理 |
|---|---|
| ffmpeg 不存在 | 路径 A 不可用；路径 B 回退 `cv2.VideoWriter`（mp4v），并在事件里记 `clip_codec=mp4v` |
| remux 切点非关键帧 | 默认接受 ≤1 GOP 偏移；`precise=true` 时重编码 |
| 源文件被删除/不可读 | 回退路径 B（如录制线程在跑）；否则记 `clip_error` |
| 队列满 | 丢弃并计数 `clip_dropped`，不阻塞主循环 |
| 写盘慢/磁盘满 | 独立线程 + 超时；失败不影响告警本身 |
| 任务隔离 | `begin_task()` 清空队列并递增代际，旧任务残留切片丢弃 |

### 1.8 验收标准

1. 本地文件源：触发告警后 ≤3s 生成切片，`ffprobe` 显示 `h264 / yuv420p`，时长 10±0.5s，**无重编码**（`-c copy` 路径可用 `ffprobe -show_frames` 对比关键帧）；
2. 流源：切片帧率 ≥ 录制帧率（≥8fps），不是 2fps 幻灯片；
3. 事件 JSON 的 `clip_path` 指向真实文件，`/api/v1/media?path=` 可播放；
4. 主循环 p95 延迟增量 <5ms（异步队列）；
5. 队列满/磁盘满/ffmpeg 崩溃三种故障下，告警与推流不受影响。

### 1.9 工作量与风险

| 项 | 估算 | 状态 |
|---|---|---|
| 路径 A（remux + 精确回退） | 0.5 天 | ✅ 已实现（实测见 §1.10） |
| 路径 B（旁路录制 + 编码） | 1–1.5 天 | ⬜ 未接线；`encode_frames()` 已备 |
| 异步队列 + 事件回写 + 存储治理 | 0.5 天 | ✅ 已实现 |

### 1.10 实现记录与实测（2026-11-19）

**新增/改动**：

| 文件 | 内容 |
|---|---|
| `src/pipeline/clip_exporter.py`（新） | `ClipRequest` / `ClipExporter`（单线程 + 有界队列 + 任务代际）/ `export_clip_file`（copy → 时长校验 → 自动重编码）/ `encode_frames`（路径 B 能力）/ `probe_duration` |
| `src/pipeline/inference_engine.py` | `clip_exporter` 初始化、`set_source_url`、`_submit_clip`、`begin_task` 代际、告警与规则命中两处接线、`metrics.scene_debug.clip_export` |
| `src/api/rest_api.py` | 任务启动时 `engine.set_source_url(stream_url)`（render 与流任务两处） |
| `configs/model_config.yaml` | `clip_export` 段（enabled/before/after/precise/max_queue/keep_days/subdir） |
| `tools/test_clip_export.py`（新） | 端到端测试：文件源切片、事件回写、流源失败、队列满 |

**实测数据**：

| 场景 | 结果 |
|---|---|
| 组件测试（laying_corner_1-1，3s+3s） | `h264 / yuv420p / 1920×1080 / 6.0s`；`clip_path` 正确回写 |
| 真实任务（high_accuracy 30s，规则告警 ×3） | 3 个切片，均 `h264 / yuv420p / 1080p / 10.0s`；事件 `alert.clip_path` 全部回写 |
| 流源（rtsp://） | `clip_error=stream_source_not_recorded`，不生成文件、不影响告警 |
| 队列打满（max_queue=2） | `dropped` 计数递增，不阻塞 |
| 失败兜底 | 切片失败只写 `clip_error`，告警与推流不受影响 |

**实测踩坑（已修）**：

1. `-t` 放在 `-i` 前被当作"输入读取时长"，配合关键帧偏移产出超长片段（6s 窗口 → 9s 文件）→ `-t` 移到 `-i` 后（输出侧）；
2. 该视频 GOP ≈ 9s，`-c copy` 只能按关键帧边界切，物理时长仍不可控 → **copy 后 `probe_duration` 校验，偏差 > max(0.8s, 25%) 自动重编码精修**；
3. 默认 `precise: true`（直接重编码，10s 1080p 约 1–2s，异步无感）；`precise: false` 走"先 copy 再按需重编"的极速路径。

**当前边界**：路径 B（RTSP 流）仍不可用——需要旁路录制线程或 ffmpeg segment 落盘；未接线前如实返回 `stream_source_not_recorded`。

---

## 2. render 编码修复（已落地）

### 2.1 问题

`run_render_task` 用 `cv2.VideoWriter(..., fourcc("mp4v"))` 输出 MP4：

- `mp4v` = MPEG-4 Part 2，**Chrome/Firefox 不支持**，部分播放器也不识别；
- 无 `faststart`，需整文件下载后才能播；
- 码率/质量不可控。

评审若在线查看 render 结果，可能直接播不了。

### 2.2 方案

新增 `src/pipeline/video_writer.py`，提供与 `cv2.VideoWriter` 同形的接口（`isOpened/write/release`）：

- 默认后端 ffmpeg：`rawvideo(bgr24) → libx264 / yuv420p / +faststart`；
- ffmpeg 不可用或启动失败 → 回退 `cv2.VideoWriter(mp4v)`，并在任务信息里记 `codec`；
- `write()` 尺寸不匹配时自动 resize；`BrokenPipe` 抛错交给 render 的 `except` 标 failed，避免"任务成功但文件坏"；
- `release()` 关闭 stdin → `wait(30s)` → 超时 kill，防僵尸进程。

**ffmpeg 必须做能力探测（实测踩坑）**：不能直接 `shutil.which("ffmpeg")`——conda/venv 里常带无 libx264 的精简版（本项目 `VdAnalyst` 环境实测 `envs/VdAnalyst/bin/ffmpeg` 仅 432KB，`-encoders` 无 libx264，进程启动即退出 → 管道 `BrokenPipe`，表现为"所有 render/切片都失败"）。探测顺序：

```
FFMPEG_BIN 环境变量 → PATH 的 ffmpeg → /usr/bin/ffmpeg → /usr/local/bin/ffmpeg → /opt/homebrew/bin/ffmpeg
逐个执行 `ffmpeg -encoders`，确认含 "libx264"；结果模块级缓存；全不支持 → 回退 cv2
```

`run_render_task` 仅改三处：构造 writer、`isOpened` 检查、finally 释放 + `last_error` 兜底；`TASKS[task_id]["render"]["codec"]` 暴露实际编码；finally 里若 `writer.last_error` 非空且状态为 finished，则改标 failed（防"文件坏但任务成功"）。

### 2.3 验收（已实测通过）

```bash
# 1) 触发 render（短片段）
curl -X POST localhost:8000/api/v1/analyze/render \
  -H 'Content-Type: application/json' \
  -d '{"stream_url":"<short>.avi","duration":5}'
# 2) 检查编码与封装
ffprobe -v error -show_entries stream=codec_name,pix_fmt -of default=nw=1 results/rendered/<task>.mp4
# 实测：codec_name=h264, pix_fmt=yuv420p, 30fps, 1.6s
# 3) 检查 faststart（moov 在 mdat 前）
python3 -c "d=open(f,'rb').read(131072); print(d.find(b'moov') < d.find(b'mdat'))"
# 实测：moov@36 < mdat@1408 → True
# 4) 完整解码校验
ffmpeg -v error -i results/rendered/<task>.mp4 -f null -
# 实测：退出码 0，无错误输出
```

回滚：`open_video_writer(..., backend="cv")` 或改回 `cv2.VideoWriter` 即恢复旧行为。

---

## 3. 多路并发（待展开）

**现状**：REST 每次任务 `asyncio.create_task(run_task)`，任务内三线程（reader/infer/encode）；但 `engine` 是**全局单例**（`rest_api.py:80`），`begin_task()` 会重置 frame_buffer / 叙述器 / 规则 / 备忘（`rest_api.py:204,296`）——两路同时跑会互相清空状态。

**设计要点（未实现，勿在材料中称"已设计"）**：
- `EnginePool`：每路一个 `InferenceEngine` 实例，per-track 状态全部实例化；
- `begin_task` 语义改为 per-engine；VLM 并发槽位跨路共享（`VLMRouter` 已是全局并发，可复用）；
- 资源预算：每路 ORT session（CUDA/CPU）+ 共享 vLLM；建议先做 2 路上限；
- 验收：两路同时跑 10 分钟，标签/事件不串、显存不涨、p95 不劣化 20% 以上；
- 工作量：3–5 天（含状态隔离测试）。

---

## 4. API 安全（待展开）

**现状**：认证/加密/脱敏仍未实现（无 `APIKey`/`HTTPBearer`/`Depends`/中间件）；WS 明文；截图与推流帧未脱敏。

**已修复（2026-11-19）：`/api/v1/media` 路径白名单**

| 项 | 规则 |
|---|---|
| 目录白名单 | `BASE_DIR`（项目根）与 `BASE_DIR.parent/data`；可用环境变量 `CSA_MEDIA_ROOTS`（`os.pathsep` 分隔）追加 |
| 扩展名白名单 | `.mp4/.avi/.mov/.mkv/.flv/.wmv/.m4v` + `.jpg/.jpeg/.png/.webp` |
| 符号链接 | `Path.resolve()` 后校验，防白名单外链接 |
| 判定顺序 | 先 403（越权）后 404（不存在），避免用状态码差异探测文件存在性 |

实测：`/etc/passwd` → 403、`configs/model_config.yaml` → 403、`../data/*.mp4` → 200、不存在的 jpg → 404。

**剩余设计要点（未实现）**：
1. **认证**：FastAPI `Depends(verify_token)` + `configs/security.yaml`（token 列表/有效期），或反代层鉴权；工作量 0.5 天；
2. **传输加密**：uvicorn `--ssl-keyfile/--ssl-certfile` 或 nginx 终结 TLS；代码零改动；
3. **人脸脱敏**：三个出口统一处理——告警截图落盘（`alert_system.py`）、`/api/v1/media`、WS 帧推送；复用 ViTPose 68 面部点做区域模糊；工作量 1–2 天；
4. **验收**：未带 token 全部端点 401；脱敏后截图人脸区域不可识别（人工抽检 20 张）；HTTPS 下 WS 用 `wss://`。

---

## 附录：ffmpeg 命令速查

```bash
# 无损切片（快速定位，切点可能提前 ≤1 GOP）
ffmpeg -ss 00:01:30 -t 10 -i in.mp4 -c copy -avoid_negative_ts make_zero out.mp4

# 精确切片（重编码对齐）
ffmpeg -ss 00:01:30 -t 10 -i in.mp4 -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p out.mp4

# 帧序列/管道编码（浏览器可播）
ffmpeg -f rawvideo -pix_fmt bgr24 -s 1920x1080 -r 25 -i pipe:0 \
  -an -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart out.mp4

# 检查
ffprobe -v error -show_entries stream=codec_name,pix_fmt,width,height -of default=nw=1 out.mp4
```
