#!/usr/bin/env bash
# Copyright 2025-2026 Dimensional Inc.
# Licensed under the Apache License, Version 2.0
#
# One-shot setup for the macOS Apple Silicon splat-camera renderer.
#
# Clones https://github.com/ghif/splat-apple into ~/.cache/dimos/splat-apple,
# applies the dimos-local cull patch (scripts/splat-apple/projection-cull.patch),
# builds the C++ Metal kernel, and exposes the package to the active venv via
# a .pth file.  Idempotent — safe to re-run.
#
# Prerequisites:
#   * macOS Apple Silicon (arm64).
#   * Active dimos venv (export VIRTUAL_ENV).
#   * `dimos[splat-mac]` extra installed (provides mlx).
#
# Usage:
#   uv pip install -e '.[splat-mac]'
#   ./scripts/install-splat-apple.sh
#   DIMOS_MLX_RASTERIZER=cpp dimos run unitree-g1-groot-wbc-sim

set -euo pipefail

# --- locations ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIMOS_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_FILE="$SCRIPT_DIR/splat-apple/projection-cull.patch"

SPLAT_APPLE_DIR="${HOME}/.cache/dimos/splat-apple"
SPLAT_APPLE_REPO="https://github.com/ghif/splat-apple"
# Pinned upstream commit the patch was authored against.  If upstream moves
# and the patch stops applying, re-pin and rebase the patch.
SPLAT_APPLE_COMMIT="bdc574338446b8394e4817b3548e75f25d463176"

# --- pretty printers ----------------------------------------------------------
if [[ -t 1 ]]; then
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    GREEN=""; YELLOW=""; RED=""; BOLD=""; RESET=""
fi
info()  { printf "%s▸%s %s\n" "$GREEN" "$RESET" "$*"; }
warn()  { printf "%s▸%s %s\n" "$YELLOW" "$RESET" "$*"; }
fail()  { printf "%s✗%s %s\n" "$RED" "$RESET" "$*" >&2; exit 1; }

# --- preflight ----------------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || fail "This script targets macOS only."
[[ "$(uname -m)" == "arm64"  ]] || fail "MlxBackend requires Apple Silicon (arm64).  Got $(uname -m)."
[[ -n "${VIRTUAL_ENV:-}" ]] || fail "No active virtualenv.  Activate the dimos venv first."

PYTHON="${VIRTUAL_ENV}/bin/python"
[[ -x "$PYTHON" ]] || fail "Active venv is missing python: $PYTHON"

PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE_PACKAGES="${VIRTUAL_ENV}/lib/python${PYVER}/site-packages"
[[ -d "$SITE_PACKAGES" ]] || fail "site-packages not found at $SITE_PACKAGES"

"$PYTHON" -c 'import mlx.core' 2>/dev/null \
    || fail "mlx not importable in venv.  Run \`uv pip install -e '.[splat-mac]'\` first."

[[ -f "$PATCH_FILE" ]] || fail "Patch file missing: $PATCH_FILE"

# --- 1. clone or update -------------------------------------------------------
mkdir -p "$(dirname "$SPLAT_APPLE_DIR")"
if [[ ! -d "$SPLAT_APPLE_DIR/.git" ]]; then
    info "cloning splat-apple to $SPLAT_APPLE_DIR"
    git clone "$SPLAT_APPLE_REPO" "$SPLAT_APPLE_DIR"
else
    info "splat-apple already cloned at $SPLAT_APPLE_DIR"
fi

# --- 2. pin to known-good commit ---------------------------------------------
cd "$SPLAT_APPLE_DIR"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$SPLAT_APPLE_COMMIT" ]]; then
    info "pinning to commit $SPLAT_APPLE_COMMIT"
    git fetch --quiet
    # Discard any stray local changes (e.g. a previously-applied patch) before
    # checking out the pinned commit.
    git checkout --quiet -- .
    git checkout --quiet "$SPLAT_APPLE_COMMIT"
fi

# --- 3. apply cull patch ------------------------------------------------------
if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    info "cull patch already applied"
elif git apply --check "$PATCH_FILE" 2>/dev/null; then
    info "applying cull patch"
    git apply "$PATCH_FILE"
else
    fail "Patch does not apply cleanly to splat-apple@$SPLAT_APPLE_COMMIT.  Re-pin or rebase the patch."
fi

# --- 4. build deps + C++ Metal kernel -----------------------------------------
info "ensuring nanobind is available (build dep)"
uv pip install --quiet 'nanobind>=2.11.0,<3'

KERNEL_SO="mlx_gs/renderer/_rasterizer_metal.$("$PYTHON" -c 'import sys; print(f"cpython-{sys.version_info.major}{sys.version_info.minor}-darwin")').so"
if [[ -f "$KERNEL_SO" ]]; then
    info "Metal C++ kernel already built ($KERNEL_SO)"
else
    info "building Metal C++ kernel — first time may take ~30s"
    "$PYTHON" setup_mlx.py build_ext --inplace > /tmp/dimos-splat-apple-build.log 2>&1 \
        || fail "build_ext failed; see /tmp/dimos-splat-apple-build.log"
fi

# --- 5. expose the package to the active venv --------------------------------
PTH_FILE="$SITE_PACKAGES/dimos_splat_apple.pth"
echo "$SPLAT_APPLE_DIR" > "$PTH_FILE"
info "wrote $PTH_FILE -> $SPLAT_APPLE_DIR"

# --- 6. verify ----------------------------------------------------------------
info "verifying import paths"
"$PYTHON" - <<'PYEOF' || fail "verification failed"
from mlx_gs.renderer.renderer import render
from mlx_gs.renderer import rasterizer_metal
assert rasterizer_metal.rasterizer_metal is not None, "C++ Metal kernel missing"
print("  ✓ mlx_gs.renderer.renderer.render importable")
print("  ✓ Metal C++ kernel loaded")
PYEOF

printf "\n%sDone.%s  Run with: %sDIMOS_MLX_RASTERIZER=cpp dimos run unitree-g1-groot-wbc-sim%s\n" \
    "$BOLD" "$RESET" "$BOLD" "$RESET"
