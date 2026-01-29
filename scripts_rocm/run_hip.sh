#!/usr/bin/env bash
set -euo pipefail

# Run FARGO3D HIP executable on GPU 0
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HIP_VISIBLE_DEVICES=0
export ROCM_VISIBLE_DEVICES=0
export HSA_VISIBLE_DEVICES=0

./bin/fargo3d_hip -D 0 setups/fargo/fargo.par > run.log 2>&1 &

echo "Started: ./bin/fargo3d_hip -D 0 setups/fargo/fargo.par"
echo "Log: $ROOT_DIR/run.log"
