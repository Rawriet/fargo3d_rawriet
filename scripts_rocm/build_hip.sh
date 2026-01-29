#!/usr/bin/env bash
set -euo pipefail

# Build FARGO3D HIP executable (single-GPU, non-MPI)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

make -f src/makefile.hip allp SETUP=fargo PARALLEL=0 GPU=1 \
  SCRIPTSDIR=./scripts_rocm SETUPSDIR=./setups STDDIR=./std SRCDIR=./src ARCHDIR=./arch BINDIR=./bin \
  EXENAME=./bin/fargo3d_hip LINKER='gcc -no-pie' \
  CUDAOPT='-include ./scripts_rocm/cuda_compat.h'
