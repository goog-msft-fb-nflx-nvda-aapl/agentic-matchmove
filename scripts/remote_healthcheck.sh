#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname
date

echo "== memory =="
free -h

echo "== swap =="
swapon --show || true

echo "== disk =="
df -h "$HOME" /tmp

echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo "== python =="
command -v python3 || true
python3 --version || true

