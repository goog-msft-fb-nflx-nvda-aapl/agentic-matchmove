#!/usr/bin/env bash
set -euo pipefail

MIN_MEM_GB="${MIN_MEM_GB:-64}"
MIN_DISK_GB="${MIN_DISK_GB:-50}"
MAX_SWAP_USED_PCT="${MAX_SWAP_USED_PCT:-25}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-0}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-5}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-1024}"

mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
mem_available_gb="$((mem_available_kb / 1024 / 1024))"

swap_total_kb="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
swap_free_kb="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
if [[ "$swap_total_kb" -gt 0 ]]; then
  swap_used_pct="$(((swap_total_kb - swap_free_kb) * 100 / swap_total_kb))"
else
  swap_used_pct=0
fi

disk_available_gb="$(df -BG "$HOME" | awk 'NR==2 {gsub("G","",$4); print $4}')"

idle_gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS=',' read -r idx util mem_used mem_total; do
    idx="${idx// /}"
    util="${util// /}"
    mem_used="${mem_used// /}"
    if [[ "$util" -le "$MAX_IDLE_GPU_UTIL" && "$mem_used" -le "$MAX_IDLE_GPU_MEM_MB" ]]; then
      idle_gpu_count="$((idle_gpu_count + 1))"
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
fi

echo "mem_available_gb=$mem_available_gb threshold=$MIN_MEM_GB"
echo "swap_used_pct=$swap_used_pct threshold=$MAX_SWAP_USED_PCT"
echo "disk_available_gb=$disk_available_gb threshold=$MIN_DISK_GB home=$HOME"
echo "idle_gpu_count=$idle_gpu_count required=$REQUIRE_IDLE_GPU"

if [[ "$mem_available_gb" -lt "$MIN_MEM_GB" ]]; then
  echo "FAIL: not enough available memory" >&2
  exit 10
fi

if [[ "$swap_used_pct" -gt "$MAX_SWAP_USED_PCT" ]]; then
  echo "FAIL: swap usage is too high" >&2
  exit 11
fi

if [[ "$disk_available_gb" -lt "$MIN_DISK_GB" ]]; then
  echo "FAIL: not enough disk space under HOME" >&2
  exit 12
fi

if [[ "$REQUIRE_IDLE_GPU" -gt 0 && "$idle_gpu_count" -lt "$REQUIRE_IDLE_GPU" ]]; then
  echo "FAIL: not enough idle GPUs" >&2
  exit 13
fi

echo "SAFE"

