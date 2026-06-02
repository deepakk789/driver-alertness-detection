"""
state.py — Shared in-memory session state
==========================================
All mutable counters and history live here so every module
(inference, routes) reads/writes the same objects.
"""

from collections import deque

HISTORY_MAX_LEN = 120   # ~60 s at 2 fps

# ---- Detection counters ----
closed_counter      = 0
yawn_counter        = 0
no_face_counter     = 0
distracted_counter  = 0

# ---- Head-pose smoothing ----
smooth_yaw   = 0.0
smooth_pitch = 0.0

# ---- History & event log ----
score_history    = deque(maxlen=HISTORY_MAX_LEN)   # [{timestamp, score, alert_level}]
event_log        = []                              # [{timestamp, event_type}]
last_alert       = None
last_alert_start = None


def reset() -> None:
    """Reset all session state to initial values."""
    global closed_counter, yawn_counter, no_face_counter, distracted_counter
    global smooth_yaw, smooth_pitch, last_alert, last_alert_start

    closed_counter = yawn_counter = no_face_counter = distracted_counter = 0
    smooth_yaw = smooth_pitch = 0.0
    last_alert = last_alert_start = None
    score_history.clear()
    event_log.clear()
