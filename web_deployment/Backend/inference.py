"""
inference.py — All CV / ML inference logic
============================================
Public API:
    run_inference(base64_frame: str, session: SessionState) -> dict

All state mutations happen on the caller-supplied SessionState object,
so this function is safe to call concurrently from multiple WebSocket
connections without any locking.
"""

import base64
import logging

import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

from state import SessionState
from models import eye_model, yawn_model

logger = logging.getLogger(__name__)

# ---- MediaPipe setup ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ---- Landmark index groups (MediaPipe Face Mesh indices) ----
LEFT_EYE_FULL  = [33, 133, 160, 158, 153, 144, 70, 63, 105, 66, 107, 65, 55, 52]
RIGHT_EYE_FULL = [362, 263, 385, 387, 373, 380, 336, 296, 334, 293, 300, 295, 285, 282]
MOUTH          = [13, 14, 78, 308, 82, 87, 317, 312, 95, 88, 178, 87, 318, 324, 402, 317]

# ---- Thresholds ----
NO_FACE_THRESHOLD          = 3
DISTRACTED_FRAME_THRESHOLD = 3
YAW_THRESHOLD              = 12
PITCH_THRESHOLD            = 12
SMOOTHING_FACTOR           = 0.8   # higher = faster response at low fps


# ============================================================
# Private helpers
# ============================================================

def _crop_region(frame, landmarks, indices: list, padding: int = 10):
    """Crop a bounding box around a set of facial landmark indices."""
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx   = (min(xs) + max(xs)) // 2
    cy   = (min(ys) + max(ys)) // 2
    half = (max(max(xs) - min(xs), max(ys) - min(ys)) // 2) + padding
    return frame[max(0, cy - half):min(h, cy + half),
                 max(0, cx - half):min(w, cx + half)]


def _head_pose(landmarks_obj, frame_shape) -> tuple[float, float]:
    """
    Estimate yaw and pitch from 6 stable facial landmarks using
    OpenCV's solvePnP (PnP = Perspective-n-Point).

    Returns (yaw_degrees, pitch_degrees).
    """
    h, w, _ = frame_shape
    face_2d, face_3d = [], []
    for idx, lm in enumerate(landmarks_obj.landmark):
        if idx in [1, 33, 263, 61, 291, 199]:   # nose, eyes, chin corners, forehead
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal   = float(w)
    cam_mat = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    dist    = np.zeros((4, 1), dtype=np.float64)

    _, rvec, _ = cv2.solvePnP(face_3d, face_2d, cam_mat, dist)
    rmat, _    = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[1] * 360, angles[0] * 360   # yaw, pitch


# ============================================================
# Public API
# ============================================================

def run_inference(base64_frame: str, session: SessionState) -> dict:
    """
    Decode a base64 JPEG frame, run the full three-pipeline inference
    (eye state, yawn state, head pose), update the supplied session
    state, and return a result dictionary.

    Parameters
    ----------
    base64_frame : str
        Raw base64-encoded JPEG data (no data-URI prefix).
    session : SessionState
        The caller's isolated session object. All counter mutations
        happen here — no global state is touched.

    Returns
    -------
    dict
        eye_status, yawn_status, head_status, probabilities,
        drowsiness_score, alert_level, timestamp.
        On decode failure: {"error": "<reason>"}.
    """
    # 1. Decode frame
    try:
        img_data = base64.b64decode(base64_frame)
    except Exception:
        return {"error": "Base64 decode failed"}

    frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Frame could not be decoded as an image"}

    # Defaults
    eye_text = yawn_text = head_text = ""
    eye_pred = yawn_pred = head_pred = 0.0

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        lm_obj    = results.multi_face_landmarks[0]
        landmarks = lm_obj.landmark

        # ---- 2. Head pose (solvePnP) ----
        raw_yaw, raw_pitch = _head_pose(lm_obj, frame.shape)
        session.smooth_yaw   = raw_yaw   * SMOOTHING_FACTOR + session.smooth_yaw   * (1 - SMOOTHING_FACTOR)
        session.smooth_pitch = raw_pitch * SMOOTHING_FACTOR + session.smooth_pitch * (1 - SMOOTHING_FACTOR)

        if abs(session.smooth_yaw) > YAW_THRESHOLD or abs(session.smooth_pitch) > PITCH_THRESHOLD:
            head_text = "AWAY"
            session.distracted_counter += 1
        else:
            head_text = "FORWARD"
            session.distracted_counter = 0

        head_pred = max(0.0, 1.0 - abs(session.smooth_yaw) / 50.0)

        # ---- 3. Eye & yawn CNN models ----
        try:
            l_eye = _crop_region(frame, landmarks, LEFT_EYE_FULL,  padding=10)
            r_eye = _crop_region(frame, landmarks, RIGHT_EYE_FULL, padding=10)
            mouth = _crop_region(frame, landmarks, MOUTH,           padding=4)

            def _prep_eye(crop):
                g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (96, 96))
                return cv2.merge([g, g, g]) / 255.0

            lp = float(eye_model.predict(np.expand_dims(_prep_eye(l_eye), 0), verbose=0)[0][0])
            rp = float(eye_model.predict(np.expand_dims(_prep_eye(r_eye), 0), verbose=0)[0][0])
            eye_pred  = (lp + rp) / 2.0
            yawn_pred = float(yawn_model.predict(
                np.reshape(cv2.resize(mouth, (96, 96)) / 255.0, (1, 96, 96, 3)),
                verbose=0)[0][0])

            eye_text  = "CLOSED" if eye_pred  < 0.5 else "OPEN"
            yawn_text = "YAWN"   if yawn_pred > 0.5 else "NOT YAWN"

            session.closed_counter  = session.closed_counter + 1 if eye_text  == "CLOSED" else 0
            session.yawn_counter    = session.yawn_counter   + 1 if yawn_text == "YAWN"   else 0
            session.no_face_counter = 0

        except Exception as e:
            logger.warning("Landmark crop/predict failed: %s", e)

    else:
        session.no_face_counter  += 1
        session.closed_counter    = 0
        session.yawn_counter      = 0

    # ---- 4. Compute alert level ----
    drowsiness_score = (session.closed_counter * 8) + (session.yawn_counter * 2)

    if session.no_face_counter >= NO_FACE_THRESHOLD:
        alert_level = "DISTRACTED"
        head_text = eye_text = yawn_text = ""
    elif session.distracted_counter >= DISTRACTED_FRAME_THRESHOLD:
        alert_level = "DISTRACTED"
    elif drowsiness_score >= 24:
        alert_level = "DROWSY"
    else:
        alert_level = "ALERT"

    # ---- 5. Record events ----
    now = datetime.now().isoformat(timespec="seconds")
    if alert_level != "ALERT" and alert_level != session.last_alert:
        session.last_alert       = alert_level
        session.last_alert_start = now
        session.event_log.append({"timestamp": now, "event_type": alert_level})
    if alert_level == "ALERT":
        session.last_alert = None

    session.score_history.append({
        "timestamp": now, "score": drowsiness_score, "alert_level": alert_level
    })

    return {
        "eye_status":         eye_text,
        "yawn_status":        yawn_text,
        "head_status":        head_text,
        "eye_prob":           round(eye_pred, 3),
        "yawn_prob":          round(yawn_pred, 3),
        "head_prob":          round(head_pred, 3),
        "drowsiness_score":   drowsiness_score,
        "distracted_counter": session.distracted_counter,
        "alert_level":        alert_level,
        "timestamp":          now,
    }
