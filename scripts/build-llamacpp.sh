#!/usr/bin/env bash
# Build the vendored llama.cpp fork (parity oracle + llama-quantize).
# cmake comes from an ephemeral nix shell (Homebrew here is nix-managed);
# nix's cmake skips Apple SDK autodetection, so the sysroot and framework
# path are passed explicitly. BLAS is off — Metal does the real work.
set -euo pipefail
cd "$(dirname "$0")/../reference/llama.cpp-laguna-branch"

SDK="$(xcrun --show-sdk-path)"
nix shell nixpkgs#cmake --command bash -c "
  cmake -B build -DGGML_METAL=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_SYSROOT='$SDK' \
    -DCMAKE_FRAMEWORK_PATH='$SDK/System/Library/Frameworks' &&
  cmake --build build -j
"
