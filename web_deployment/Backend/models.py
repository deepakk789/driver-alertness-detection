"""
models.py — Load trained Keras models in a background thread
=============================================================
Models are loaded asynchronously so uvicorn can bind its port
immediately. Without this, TensorFlow's slow startup blocks the
process for 2-3 minutes, causing Render's health-check to time out
and mark the deployment as failed.

Usage anywhere in the backend:
    from models import eye_model, yawn_model, models_ready
    if not models_ready.is_set(): ...  # still loading
"""

import os
import threading
import logging

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EYE_MODEL  = os.path.join(BASE_DIR, "Models", "eye_model_mobilenet_tuned.h5")
YAWN_MODEL = os.path.join(BASE_DIR, "Models", "yawn_model_mobilenet_tuned.h5")

# Placeholders — set by _load() once the thread completes
eye_model  = None
yawn_model = None

# Event that other modules can wait on / check
models_ready = threading.Event()


def _load():
    """Load both Keras models; runs in a daemon thread."""
    global eye_model, yawn_model
    try:
        from tensorflow.keras.models import load_model   # import here to keep startup fast
        logger.info("[models] Loading eye model...")
        eye_model  = load_model(EYE_MODEL)
        logger.info("[models] Loading yawn model...")
        yawn_model = load_model(YAWN_MODEL)
        models_ready.set()
        logger.info("[models] Both models loaded and ready.")
    except Exception as e:
        logger.error("[models] Failed to load models: %s", e)
        # Even on failure, set the event so waiters don't block forever
        models_ready.set()


# Kick off loading immediately when this module is imported,
# but don't block — uvicorn will bind the port while this runs.
_thread = threading.Thread(target=_load, daemon=True, name="model-loader")
_thread.start()
