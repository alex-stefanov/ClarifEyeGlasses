"""
Download and prepare SmolVLM-256M-Instruct for ClarifEye scene description.

MODEL SIZE: ~500 MB total (ONNX int8 export + tokenizer files).
Do NOT commit these files to git — download on the Pi or scp from desktop.

EXPORT STRATEGY
~~~~~~~~~~~~~~~
SmolVLM-256M is a generative vision-language model (encoder + autoregressive
decoder).  A single flat .onnx file cannot represent an autoregressive
generation loop, so we use optimum-cli which exports the model as a set of
ONNX files in a directory (encoder.onnx, decoder_model.onnx, etc.) and
handles the generation loop via ORTModelForVision2Seq.

The export directory is written to:
  models/smolvlm_256m_int8.onnx/   (despite the .onnx extension, this is a directory)
The tokenizer/processor is saved separately to:
  models/smolvlm_256m_tokenizer/

Primary path  (recommended):
  1. pip install optimum[onnxruntime] onnxruntime
  2. This script runs optimum-cli export to produce the int8 ONNX directory.
  Estimated time: 10–20 min on Pi 4 (or 2–5 min on desktop + scp).

Fallback path (if optimum export fails):
  The script downloads the raw transformers checkpoint instead.
  SceneDescriber will use it via PyTorch CPU inference (~15–30 s/frame on Pi).
  To trigger the fallback manually: run with --fallback

Run on the Pi (internet required, ~500 MB download):
  pip install optimum[onnxruntime] onnxruntime transformers Pillow
  python scripts/download_smolvlm.py
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config import MODELS_DIR, SCENE_MODEL_PATH, SCENE_TOKENIZER_PATH  # noqa: E402

HF_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _size_mb(path: str) -> float:
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / 1_048_576
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1_048_576


def export_onnx_optimum() -> bool:
    """Export SmolVLM to ONNX int8 using optimum-cli."""
    try:
        import optimum  # noqa: F401
    except ImportError:
        print("optimum not installed. Install with: pip install optimum[onnxruntime]")
        return False

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Exporting {HF_MODEL_ID} to ONNX int8 …")
    print(f"Output directory: {SCENE_MODEL_PATH}")
    print("This may take 10–30 minutes on a Raspberry Pi 4.")
    print()

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", HF_MODEL_ID,
        "--task", "image-to-text-with-past",
        "--quantize",
        "--optimum_quantize_config", "arm64",
        SCENE_MODEL_PATH,
    ]

    # Simpler fallback command if the above flags are not supported.
    cmd_simple = [
        "optimum-cli", "export", "onnx",
        "--model", HF_MODEL_ID,
        "--task", "image-to-text-with-past",
        SCENE_MODEL_PATH,
    ]

    for attempt_cmd in (cmd, cmd_simple):
        print(f"Running: {' '.join(attempt_cmd)}")
        result = subprocess.run(attempt_cmd)
        if result.returncode == 0:
            break
        print(f"Command failed (exit {result.returncode}), trying alternative …")
    else:
        print("ONNX export failed via both command variants.")
        return False

    # Save the processor/tokenizer to the separate tokenizer directory.
    print(f"\nSaving tokenizer to: {SCENE_TOKENIZER_PATH}")
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(HF_MODEL_ID)
        proc.save_pretrained(SCENE_TOKENIZER_PATH)
        print("Tokenizer saved.")
    except Exception as exc:
        print(f"Tokenizer save failed: {exc}")
        print(
            f"Manual fix: copy the processor/tokenizer from the optimum export "
            f"directory at {SCENE_MODEL_PATH} into {SCENE_TOKENIZER_PATH}"
        )

    size = _size_mb(SCENE_MODEL_PATH)
    print(f"\nModel directory size: {size:.0f} MB")
    return True


def download_transformers_fallback() -> bool:
    """
    Download the raw HuggingFace checkpoint (PyTorch weights).

    SceneDescriber will use transformers CPU inference when the ONNX export is absent.
    Inference latency on Pi 4 will be 15–30 s/frame instead of 5–10 s.
    """
    print(f"Downloading {HF_MODEL_ID} as transformers checkpoint (fallback) …")
    print(
        "Inference with this path will use PyTorch CPU (~15–30 s/frame on Pi 4)."
    )
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        os.makedirs(SCENE_MODEL_PATH, exist_ok=True)
        os.makedirs(SCENE_TOKENIZER_PATH, exist_ok=True)

        print("Downloading processor …")
        proc = AutoProcessor.from_pretrained(HF_MODEL_ID)
        proc.save_pretrained(SCENE_TOKENIZER_PATH)

        print("Downloading model weights (this will take several minutes) …")
        model = AutoModelForVision2Seq.from_pretrained(HF_MODEL_ID)
        model.save_pretrained(SCENE_MODEL_PATH)

        size = _size_mb(SCENE_MODEL_PATH)
        print(f"\nModel checkpoint size: {size:.0f} MB")
        print(f"Saved to: {SCENE_MODEL_PATH}")
        return True
    except Exception as exc:
        print(f"Fallback download failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SmolVLM-256M for ClarifEye")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Skip ONNX export, download raw transformers checkpoint instead.",
    )
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(SCENE_MODEL_PATH):
        size = _size_mb(SCENE_MODEL_PATH)
        print(f"Model already exists at {SCENE_MODEL_PATH} ({size:.0f} MB). Skipping.")
        return

    if args.fallback:
        ok = download_transformers_fallback()
    else:
        ok = export_onnx_optimum()
        if not ok:
            print("\nONNX export failed — attempting transformers fallback download …")
            ok = download_transformers_fallback()

    if ok:
        print("\nSmolVLM download/export complete.")
        print("Scene description mode is ready.")
    else:
        print("\nAll download attempts failed.")
        print("Manual option: on a desktop with more RAM, run:")
        print(f"  optimum-cli export onnx --model {HF_MODEL_ID} "
              f"--task image-to-text-with-past {SCENE_MODEL_PATH}")
        print("Then scp the directory to the Pi.")
        sys.exit(1)


if __name__ == "__main__":
    main()
