from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from solidcue.observability import get_env_path

HHEM_MODEL_ID = "vectara/hallucination_evaluation_model"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "hhem"

_model = None
load_dotenv(dotenv_path=get_env_path())


def _get_device() -> str:
    import torch

    configured_device = os.getenv("SOLIDCUE_HHEM_DEVICE", "").strip().lower()
    if configured_device in {"cpu", "mps"}:
        return configured_device

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hhem_model():
    global _model
    if _model is not None:
        return _model

    from transformers import AutoModelForSequenceClassification

    model_path = str(MODELS_DIR)
    if MODELS_DIR.exists() and (MODELS_DIR / "config.json").exists():
        _model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _model = AutoModelForSequenceClassification.from_pretrained(HHEM_MODEL_ID, trust_remote_code=True)
        _model.save_pretrained(model_path)
    _model = _model.to(_get_device())
    device = next(_model.parameters()).device
    print(f"HHEM model loaded: {HHEM_MODEL_ID} | device: {device}")
    return _model


def get_hhem_model():
    if _model is None:
        return load_hhem_model()
    return _model
