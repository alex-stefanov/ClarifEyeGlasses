"""
ClarifEye Scene Describer

Wraps SmolVLM-256M-Instruct to generate a one-sentence description of a
camera frame. Two inference backends are tried in order:

  1. optimum.onnxruntime.ORTModelForVision2Seq  — ONNX-accelerated, preferred.
     Requires the model exported by scripts/download_smolvlm.py.
     config.SCENE_MODEL_PATH should be a directory produced by optimum-cli.

  2. transformers AutoModelForVision2Seq (PyTorch CPU)  — fallback when
     optimum is not installed or the ONNX export is absent.

Realistic latency on Pi 4: 5–15 s per inference. The caller should play the
"processing" audio cue before invoking describe() so the user is not confused
by the silence.

Returns None (never raises) so the caller can degrade gracefully.
"""
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("clarifeye.ai.scene_describer")

_PROMPT = "Describe this image briefly in one sentence."


class SceneDescriber:
    """
    Single-shot scene description using SmolVLM-256M-Instruct.

    Construction does NOT crash on missing model files — it logs a warning and
    marks the instance as not ready. describe() returns None immediately when
    not ready.
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        self._ready = False
        self._model = None
        self._processor = None
        self._backend: str = "none"

        model_exists = os.path.exists(model_path)
        tokenizer_exists = os.path.exists(tokenizer_path)

        if not model_exists:
            logger.warning(
                "SmolVLM model not found at '%s'. "
                "Run scripts/download_smolvlm.py on the Pi to download it. "
                "Scene mode will be unavailable.",
                model_path,
            )
            return
        if not tokenizer_exists:
            logger.warning(
                "SmolVLM tokenizer not found at '%s'. "
                "Run scripts/download_smolvlm.py on the Pi to download it. "
                "Scene mode will be unavailable.",
                tokenizer_path,
            )
            return

        self._load(model_path, tokenizer_path)

    def _load(self, model_path: str, tokenizer_path: str) -> None:
        # ── Try optimum ONNX first ────────────────────────────────────────────
        try:
            from optimum.onnxruntime import ORTModelForVision2Seq  # type: ignore[import]
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(tokenizer_path)
            self._model = ORTModelForVision2Seq.from_pretrained(model_path)
            self._backend = "optimum"
            self._ready = True
            logger.info("SceneDescriber: loaded SmolVLM via optimum-onnxruntime.")
            return
        except ImportError:
            logger.warning(
                "optimum not installed — falling back to transformers (PyTorch CPU). "
                "Install optimum[onnxruntime] for faster inference."
            )
        except Exception as exc:
            logger.warning(
                "optimum load failed (%s) — falling back to transformers.", exc
            )

        # ── Fallback: transformers PyTorch CPU ────────────────────────────────
        try:
            import torch  # type: ignore[import]
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(tokenizer_path)
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype=torch.float32
            )
            self._model.eval()
            self._backend = "transformers"
            self._ready = True
            logger.info("SceneDescriber: loaded SmolVLM via transformers (PyTorch CPU).")
        except Exception as exc:
            logger.error(
                "SceneDescriber: failed to load SmolVLM via any backend: %s. "
                "Scene mode will be unavailable.",
                exc,
            )

    def is_ready(self) -> bool:
        return self._ready

    def describe(self, frame: np.ndarray, max_words: int = 25) -> Optional[str]:
        """
        Generate a one-sentence English description of *frame*.

        Args:
            frame:     RGB numpy array from the camera.
            max_words: Hard cap on output words; sentence-boundary truncation
                       is preferred when possible.

        Returns:
            English description string, or None if inference failed.
        """
        if not self._ready:
            return None

        try:
            from PIL import Image  # type: ignore[import]

            img = Image.fromarray(frame)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ]

            prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self._processor(
                text=prompt, images=[img], return_tensors="pt"
            )

            if self._backend == "optimum":
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_words + 15
                )
            else:
                import torch  # type: ignore[import]

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs, max_new_tokens=max_words + 15
                    )

            # Decode only the newly generated tokens.
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            text = self._processor.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            return _truncate(text, max_words)

        except Exception as exc:
            logger.error(
                "SceneDescriber.describe() failed: %s", exc, exc_info=True
            )
            return None


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    # Try to end at sentence-boundary punctuation within the last 5 words.
    for i in range(min(max_words, len(words)) - 1, max(0, max_words - 5) - 1, -1):
        if words[i].endswith((".", "!", "?")):
            return " ".join(words[: i + 1])
    return " ".join(words[:max_words])
