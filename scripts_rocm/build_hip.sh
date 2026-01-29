#!/usr/bin/env bash
set -euo pipefail

# Build FARGO3D HIP executable (single-GPU, non-MPI)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Pre-generate bound_code files from the canonical scripts dir to avoid
# path issues in fresh clones (bound_code.py writes to ../scripts/*).
SETUP="${SETUP:-fargo}"
NFLUIDS="${NFLUIDS:-1}"
( if [[ ! -e "$ROOT_DIR/scripts/c2hip.py" && -e "$ROOT_DIR/scripts/c2hip_rocm.py" ]]; then
    ln -s c2hip_rocm.py "$ROOT_DIR/scripts/c2hip.py"
  fi )
( cd "$ROOT_DIR/scripts" && python3 bound_code.py "$SETUP" "$NFLUIDS" )

make -f src/makefile.hip allp SETUP=fargo PARALLEL=0 GPU=1 \
  SCRIPTSDIR=./scripts SETUPSDIR=./setups STDDIR=./std SRCDIR=./src ARCHDIR=./arch BINDIR=./bin \
  EXENAME=./bin/fargo3d_hip LINKER='gcc -no-pie' \
  CUDAOPT='-include ./scripts_rocm/cuda_compat.h'
