#!/usr/bin/env bash
# vllm_ctl.sh — Qwen2.5-VL vLLM 服务控制：start / stop / freeze / resume / restart / status
#
# 为什么需要 freeze/resume：
#   实测（2026-11-22, V100）vLLM 进程"活着但空闲"也会把检测吞吐从 ~723 帧/60s
#   打到 ~393 帧/60s；SIGSTOP 冻结（显存照常占用、进程不调度）后立刻恢复到 716 帧。
#   瓶颈是进程运行时的调度争用，不是显存、不是量化精度（4-bit 无效）。
#   所以：实时看框/演示 → freeze；需要 VLM 复核/挖掘 → resume。
#
# 用法：
#   tools/vllm_ctl.sh start      # 启动（~90s 加载，等待就绪）
#   tools/vllm_ctl.sh stop       # 停止并释放显存
#   tools/vllm_ctl.sh freeze     # 冻结（检测恢复满速，VLM 保留显存与权重）
#   tools/vllm_ctl.sh resume     # 解冻（立即可推理，无需重新加载）
#   tools/vllm_ctl.sh restart    # stop + start
#   tools/vllm_ctl.sh status     # 查看状态
#
# 环境变量可覆盖默认值：PYTHON / MODEL_DIR / SERVED_NAME / PORT / GPU_UTIL / MAX_LEN
set -u

PYTHON="${PYTHON:-$HOME/.conda/envs/VdAnalyst/bin/python}"
MODEL_DIR="${MODEL_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/Qwen2.5-VL-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen2.5-vl-7b-instruct}"
PORT="${PORT:-8001}"
GPU_UTIL="${GPU_UTIL:-0.60}"
MAX_LEN="${MAX_LEN:-12288}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${LOG:-$SCRIPT_DIR/../results/logs/vllm_${PORT}.log}"
PIDFILE="${LOG%.log}.pid"
MAX_WAIT_SEC=240

log() { echo "[vllm_ctl] $*"; }

# 主进程 pid（优先取监听端口的进程，其次 pid 文件）
main_pid() {
    local pid
    pid=$(ss -ltnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ -z "$pid" ] && [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE" 2>/dev/null | grep -E '^[0-9]+$' || true)
        [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && pid=""
    fi
    echo "$pid"
}

# 主进程 + 全部子孙进程（vLLM 用多进程引擎，冻结必须覆盖引擎 worker）
all_pids() {
    local parent="$1"
    [ -z "$parent" ] && return 0
    local out="$parent"
    local children c
    children=$(pgrep -P "$parent" 2>/dev/null || true)
    for c in $children; do
        out="$out $(all_pids "$c")"
    done
    echo $out
}

proc_state() {
    local pid="$1"
    awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || echo "?"
}

cmd_start() {
    local pid
    pid=$(main_pid)
    if [ -n "$pid" ]; then
        log "已在运行 (pid=$pid, port=$PORT)。要重启请用 restart"
        return 0
    fi
    if [ ! -d "$MODEL_DIR" ]; then
        log "模型目录不存在: $MODEL_DIR"
        return 1
    fi
    mkdir -p "$(dirname "$LOG")"
    log "启动 vLLM: model=$MODEL_DIR port=$PORT gpu_util=$GPU_UTIL max_len=$MAX_LEN"
    log "日志: $LOG"
    nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --dtype float16 \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --limit-mm-per-prompt image=4 \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    log "等待就绪（最长 ${MAX_WAIT_SEC}s）..."
    local waited=0
    while [ "$waited" -lt "$MAX_WAIT_SEC" ]; do
        if curl -sf -m 2 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            log "就绪，用时 ${waited}s。"
            cmd_status
            return 0
        fi
        if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            log "启动失败，最近日志："
            tail -15 "$LOG"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "等待超时，请查看日志: $LOG"
    return 1
}

cmd_stop() {
    local pid
    pid=$(main_pid)
    if [ -z "$pid" ]; then
        log "未发现运行中的 vLLM (port=$PORT)"
        rm -f "$PIDFILE"
        return 0
    fi
    local p
    for p in $(all_pids "$pid"); do
        kill -CONT "$p" 2>/dev/null || true   # 冻结中的进程先解冻才能正常退出
        kill -TERM "$p" 2>/dev/null || true
    done
    log "已发送 TERM (pids: $(all_pids "$pid" | tr '\n' ' '))，等待退出..."
    local waited=0
    while [ "$waited" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "未退出，发送 KILL"
        for p in $(all_pids "$pid"); do
            kill -CONT "$p" 2>/dev/null || true
            kill -KILL "$p" 2>/dev/null || true
        done
        sleep 3
    fi
    rm -f "$PIDFILE"
    log "已停止。"
}

# 冻结：显存保留、权重保留，但进程不再调度 → 检测吞吐恢复满速
cmd_freeze() {
    local pid
    pid=$(main_pid)
    if [ -z "$pid" ]; then
        log "未发现运行中的 vLLM，无法冻结。"
        return 1
    fi
    local p
    for p in $(all_pids "$pid"); do
        kill -STOP "$p" 2>/dev/null || true
    done
    sleep 1
    log "已冻结 (pids: $(all_pids "$pid" | tr '\n' ' '))。"
    log "注意：冻结期间 VLM 不响应任何请求（复核/发现会超时或退避）。"
}

cmd_resume() {
    local pid
    pid=$(main_pid)
    if [ -z "$pid" ]; then
        log "未发现 vLLM 进程。"
        return 1
    fi
    local p
    for p in $(all_pids "$pid"); do
        kill -CONT "$p" 2>/dev/null || true
    done
    sleep 1
    log "已解冻，VLM 可立即推理（无需重新加载）。"
}

cmd_status() {
    local pid
    pid=$(main_pid)
    if [ -z "$pid" ]; then
        log "状态: stopped（显存已释放）"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[vllm_ctl] GPU: /'
        return 0
    fi
    local state frozen="no"
    state=$(proc_state "$pid")
    [ "$state" = "T" ] && frozen="yes"
    log "状态: $([ "$frozen" = "yes" ] && echo "frozen（冻结中，检测满速）" || echo "running")  pid=$pid  端口=$PORT"
    local model
    if [ "$frozen" = "yes" ]; then
        model="(冻结中不响应 API)"
    else
        model=$(curl -sf -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null \
                | grep -oP '"id"\s*:\s*"\K[^"]+' | head -1)
        model="${model:-未知/未就绪}"
    fi
    log "模型: $model  (日志: $LOG)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[vllm_ctl] GPU: /'
    log "切换提示: freeze=看框演示满速  resume=恢复VLM复核/挖掘"
}

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    freeze)  cmd_freeze ;;
    resume)  cmd_resume ;;
    restart) cmd_stop; sleep 2; cmd_start ;;
    status)  cmd_status ;;
    *)
        echo "用法: $0 {start|stop|freeze|resume|restart|status}"
        exit 1
        ;;
esac
