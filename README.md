# ClarifEye Glasses

ClarifEye Glasses is a Raspberry Pi based wearable vision-assistance system. It combines camera input, distance sensors, object detection, OCR, traffic-light detection, translation, and audio output to provide real-time guidance for visually impaired users.

## Features

- Multi-mode assistant for traffic lights, navigation, text reading, currency recognition, and scene description
- Camera pipeline with low-light enhancement
- TFLite object and traffic-light detection models
- HSV color verification for traffic-light results
- Ultrasonic and ToF distance sensor fusion
- Priority engine for deciding which objects should be announced first
- Offline-oriented audio feedback with language support
- Test and benchmarking scripts for hardware and AI modules

## Tech Stack

- Python
- Raspberry Pi GPIO/I2C hardware
- TFLite models
- OpenCV-style image processing
- OCR and translation utilities
- Piper/espeak-ng style local speech output

## Project Structure

```text
clarifeye/
├── ai/          # Detection, OCR, translation, scene, and currency modules
├── core/        # Mode management, sensor fusion, settings, priority logic
├── data/        # Cached phrases, translation data, and reference assets
├── hardware/    # Camera, buttons, audio, ToF, and ultrasonic integrations
├── models/      # TFLite model assets
├── scripts/     # Model/data download and training utilities
├── tests/       # Hardware and integration tests
├── config.py
└── main.py
```

## Getting Started

### Prerequisites

- Raspberry Pi 4 or compatible Linux device
- Python 3
- Camera module
- HC-SR04 ultrasonic sensors
- VL53L0X ToF sensor
- Audio output device

### Install

```bash
git clone https://github.com/alex-stefanov/ClarifEyeGlasses.git
cd ClarifEyeGlasses/clarifeye
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download any required model and voice assets with the scripts in `clarifeye/scripts/` before running on hardware.

### Run

```bash
python main.py
```

Use `debug_main.py`, `live_preview.py`, and the tests under `tests/` when validating modules without the full wearable loop.

## Configuration

Hardware pins, model paths, thresholds, audio settings, OCR settings, and mode constants are centralized in `clarifeye/config.py`.

## Testing

```bash
pytest
```

Some tests require connected Raspberry Pi hardware and will not run correctly on a regular desktop machine.

## License

No license file is currently included.
