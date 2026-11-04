"""Text-in, class-and-confidence-out inference for the saved GRU model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from train import CLASSIFICATION_THRESHOLD


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "sms_spam_gru.keras"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the serialized Keras model, including its text vectorizer."""
    import tensorflow as tf

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el modelo entrenado en {path}. Ejecuta `python train.py` "
            "o el notebook de Colab para generarlo."
        )
    return tf.keras.models.load_model(path, compile=False)


def predict_message(message: str, model=None, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Predict from raw SMS text; all normalization/vectorization stays in-model."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Escribe un mensaje SMS antes de solicitar la predicción.")
    classifier = model if model is not None else load_model(model_path)
    import tensorflow as tf

    message_tensor = tf.convert_to_tensor([message], dtype=tf.string)
    spam_probability = float(
        np.asarray(classifier.predict(message_tensor, verbose=0)).reshape(-1)[0]
    )
    if not np.isfinite(spam_probability) or not 0.0 <= spam_probability <= 1.0:
        raise RuntimeError("El modelo produjo una probabilidad inválida.")
    predicted_spam = spam_probability >= CLASSIFICATION_THRESHOLD
    return {
        "label": "spam" if predicted_spam else "legítimo",
        "class_key": "spam" if predicted_spam else "ham",
        "confidence": spam_probability if predicted_spam else 1.0 - spam_probability,
        "spam_probability": spam_probability,
        "threshold": CLASSIFICATION_THRESHOLD,
    }
