#!/usr/bin/env bash
# Download the official Laguna S 2.1 GGUFs into models/.
# Q4_K_M is the only official quant that fits in 128GB unified memory
# (Q8_0 is 127.7GB — over the wired limit; F16 is 235GB). The DFlash draft
# model is fetched for later speculative-decoding work (see TODO.md).
set -euo pipefail
cd "$(dirname "$0")/../models"

base="https://huggingface.co/poolside/Laguna-S-2.1-GGUF/resolve/main"
for f in laguna-s-2.1-Q4_K_M.gguf laguna-s-2.1-DFlash-BF16.gguf; do
  curl -L -C - -o "$f" "$base/$f"
done
