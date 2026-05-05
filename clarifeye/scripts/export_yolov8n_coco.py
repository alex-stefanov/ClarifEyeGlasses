"""
Export YOLOv8n COCO weights to TFLite INT8 at 320x320.

RUN THIS ON A DESKTOP MACHINE — not on the Pi 4.
The Pi 4 lacks the RAM to perform this export.

Requirements:
    pip install ultralytics ai-edge-litert

Output:
    models/yolov8n_coco_int8.tflite

After export, copy the file to the Pi:
    scp models/yolov8n_coco_int8.tflite pi@<pi-ip>:~/clarifeye/clarifeye/models/

Expected output tensor shape: [1, 84, 2100]
  84   = 4 bbox coords (cx, cy, w, h in model-pixel space) + 80 COCO class scores
  2100 = 40x40 + 20x20 + 10x10 anchor grid cells across 3 strides (for 320x320 input)
If the shape arrives transposed as [1, 2100, 84], the detector handles that too.
"""
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "models", "yolov8n_coco_int8.tflite")


def main() -> None:
    if os.path.exists(OUTPUT_PATH):
        print(f"Model already exists at {OUTPUT_PATH} — skipping export.")
        _print_tensor_shape(OUTPUT_PATH)
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed.")
        print("Install it with:  pip install ultralytics")
        sys.exit(1)

    print("Downloading yolov8n.pt (if not cached) and exporting to TFLite INT8 …")
    print("This may take several minutes and requires ~4 GB RAM.\n")

    model = YOLO("yolov8n.pt")
    exported = model.export(
        format="tflite",
        imgsz=320,
        int8=True,
        data="coco.yaml",
    )

    exported_str = str(exported)
    print(f"\nExport finished: {exported_str}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if exported_str != OUTPUT_PATH:
        shutil.copy(exported_str, OUTPUT_PATH)
        print(f"Copied to:       {OUTPUT_PATH}")

    _print_tensor_shape(OUTPUT_PATH)


def _print_tensor_shape(model_path: str) -> None:
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        print("\n(ai_edge_litert not installed — cannot print tensor shapes.)")
        return

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    in_details = interp.get_input_details()
    out_details = interp.get_output_details()

    print("\nInput tensor shapes:")
    for d in in_details:
        print(f"  shape={tuple(d['shape'])}  dtype={d['dtype'].__name__}")

    print("\nOutput tensor shapes:")
    for i, d in enumerate(out_details):
        print(f"  [{i}]  shape={tuple(d['shape'])}  dtype={d['dtype'].__name__}")

    print(
        "\nExpected: [1, 84, 2100]"
        "  (84 = 4 bbox + 80 COCO classes, 2100 anchor cells at 320x320)"
    )
    print(
        "The detector in ai/object_detector.py transposes [84, N] → [N, 84] "
        "and raises an error if neither axis equals 84."
    )


if __name__ == "__main__":
    main()
