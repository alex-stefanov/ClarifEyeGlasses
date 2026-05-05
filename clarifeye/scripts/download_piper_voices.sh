#!/usr/bin/env bash
# Download Piper TTS voice models for ClarifEye.
# Models are stored in voices/ (gitignored — they are ~50 MB each).
# This script is idempotent: already-downloaded files are skipped.
#
# Usage:  bash scripts/download_piper_voices.sh
# Requires: curl

set -euo pipefail

VOICES_DIR="$(cd "$(dirname "$0")/.." && pwd)/voices"
REPO="https://huggingface.co/rhasspy/piper-voices/resolve/main"

mkdir -p "$VOICES_DIR"

download_if_missing() {
    local lang="$1"   # e.g. "en/en_US/amy/medium"
    local name="$2"   # e.g. "en_US-amy-medium"

    local onnx_url="${REPO}/${lang}/${name}.onnx"
    local json_url="${REPO}/${lang}/${name}.onnx.json"
    local onnx_path="${VOICES_DIR}/${name}.onnx"
    local json_path="${VOICES_DIR}/${name}.onnx.json"

    if [[ -f "$onnx_path" && -f "$json_path" ]]; then
        echo "[skip] ${name} already present"
        return
    fi

    echo "[download] ${name}.onnx ..."
    curl -L --progress-bar -o "$onnx_path" "$onnx_url"
    echo "[download] ${name}.onnx.json ..."
    curl -L --progress-bar -o "$json_path" "$json_url"
    echo "[done] ${name}"
}

download_if_missing "en/en_US/amy/medium"    "en_US-amy-medium"
download_if_missing "bg/bg_BG/dimitar/medium" "bg_BG-dimitar-medium"

echo ""
echo "Voices ready in: ${VOICES_DIR}"
