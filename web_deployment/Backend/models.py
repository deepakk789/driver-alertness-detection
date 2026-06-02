"""
models.py — Load trained Keras models once at server startup
=============================================================
Import `eye_model` and `yawn_model` from this module anywhere
in the backend — they are only loaded once.
"""

import os
from tensorflow.keras.models import load_model

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EYE_MODEL  = os.path.join(BASE_DIR, "Models", "eye_model_mobilenet_tuned.h5")
YAWN_MODEL = os.path.join(BASE_DIR, "Models", "yawn_model_mobilenet_tuned.h5")

print("[INFO] Loading models...")
eye_model  = load_model(EYE_MODEL)
yawn_model = load_model(YAWN_MODEL)
print("[INFO] Models loaded successfully.")
