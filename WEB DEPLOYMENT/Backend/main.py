"""
Driver Alertness Detection System — FastAPI Backend
====================================================
This server loads the 3 trained Keras models once at startup,
exposes a /predict endpoint that receives a webcam frame from
the browser, runs inference, and returns a JSON alertness result.

Models are accessed from the existing Models/ directory.
This file lives in WEB DEPLOYMENT/Backend/ and is completely
separate from the original ML training scripts in Src/.
"""

import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from collections import deque
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ============================================================
# Paths — Models accessed from the parent project directory
# ============================================================
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EYE_MODEL  = os.path.join(BASE_DIR, "Models", "eye_model_mobilenet_tuned.h5")
YAWN_MODEL = os.path.join(BASE_DIR, "Models", "yawn_model_mobilenet_tuned.h5")

# ============================================================
# Load Models (once at server startup)
# ============================================================
print("[INFO] Loading models...")
eye_model  = load_model(EYE_MODEL)
yawn_model = load_model(YAWN_MODEL)
print("[INFO] All models loaded successfully.")

# ============================================================
# MediaPipe Setup
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indexes (same as original script)
LEFT_EYE_FULL = [
    33, 133, 160, 158, 153, 144,
    70, 63, 105, 66, 107,
    65, 55, 52,
]
RIGHT_EYE_FULL = [
    362, 263, 385, 387, 373, 380,
    336, 296, 334, 293, 300,
    295, 285, 282,
]
MOUTH = [
    13, 14, 78, 308,
    82, 87, 317, 312,
    95, 88, 178, 87,
    318, 324, 402, 317
]

# ============================================================
# Temporal Smoothing State (in-memory, per-server-session)
# ============================================================
CLOSED_FRAME_THRESHOLD     = 9
YAWN_FRAME_THRESHOLD       = 10
NO_FACE_THRESHOLD          = 30
DISTRACTED_FRAME_THRESHOLD = 30
HISTORY_MAX_LEN            = 120   # keep last 120 data points (~60s at 2fps)

closed_counter    = 0
yawn_counter      = 0
no_face_counter   = 0
distracted_counter= 0

smooth_yaw        = 0.0
smooth_pitch      = 0.0

score_history  = deque(maxlen=HISTORY_MAX_LEN)  # [{timestamp, score, alert_level}]
event_log      = []                              # [{timestamp, event_type, duration}]
last_alert     = None
last_alert_start = None

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Driver Alertness API", version="1.0.0")

# Allow browser to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the Frontend folder as static files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frontend"))
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================
# Helper: Crop Region
# ============================================================
def crop_region(frame, landmarks, indices, padding=10):
    h, w, _ = frame.shape
    points   = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    box_w   = x_max - x_min
    box_h   = y_max - y_min
    max_dim = max(box_w, box_h)
    half    = (max_dim // 2) + padding
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return frame[y1:y2, x1:x2]


# ============================================================
# Request/Response Schema
# ============================================================
class FrameRequest(BaseModel):
    frame: str  # base64-encoded JPEG string


# ============================================================
# POST /predict — Main inference endpoint
# ============================================================
@app.post("/predict")
def predict(req: FrameRequest):
    global closed_counter, yawn_counter, no_face_counter, distracted_counter
    global smooth_yaw, smooth_pitch
    global last_alert, last_alert_start

    # 1. Decode base64 frame to NumPy image
    img_data = base64.b64decode(req.frame)
    np_arr   = np.frombuffer(img_data, np.uint8)
    frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid frame"}

    # ---- Default values ----
    eye_text       = ""
    yawn_text      = ""
    head_text      = ""
    eye_pred       = 0.0
    yawn_pred      = 0.0
    head_pred      = 0.0
    drowsiness_score = 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- Inference via MediaPipe ----
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks_obj = results.multi_face_landmarks[0]
        landmarks = landmarks_obj.landmark
        
        # --- 1. Robust Head Pose Estimation ---
        face_3d, face_2d = [], []
        h, w, _ = frame.shape

        for idx, lm in enumerate(landmarks_obj.landmark):
            if idx in [1, 33, 263, 61, 291, 199]:
                x, y = int(lm.x * w), int(lm.y * h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * w
        cam_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        YAW_THRESHOLD = 12
        PITCH_THRESHOLD = 12
        SMOOTHING_FACTOR = 0.1

        smooth_pitch = (angles[0] * 360 * SMOOTHING_FACTOR) + (smooth_pitch * (1 - SMOOTHING_FACTOR))
        smooth_yaw   = (angles[1] * 360 * SMOOTHING_FACTOR) + (smooth_yaw * (1 - SMOOTHING_FACTOR))

        if abs(smooth_yaw) > YAW_THRESHOLD or abs(smooth_pitch) > PITCH_THRESHOLD:
            head_text = "AWAY"
            distracted_counter += 1
        else:
            head_text = "FORWARD"
            distracted_counter = 0
            
        head_pred = max(0.0, 1.0 - (abs(smooth_yaw) / 50.0))

        # --- 2. Eye & Yawn (MediaPipe Crops) ---
        try:
            left_eye  = crop_region(frame, landmarks, LEFT_EYE_FULL,  padding=10)
            right_eye = crop_region(frame, landmarks, RIGHT_EYE_FULL, padding=10)
            mouth     = crop_region(frame, landmarks, MOUTH,           padding=4)

            # Eye (MobileNet expects 3-channel grayscale-like for MRL dataset)
            left_gray  = cv2.resize(cv2.cvtColor(left_eye,  cv2.COLOR_BGR2GRAY), (96, 96))
            right_gray = cv2.resize(cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY), (96, 96))
            left_img   = cv2.merge([left_gray, left_gray, left_gray]) / 255.0
            right_img  = cv2.merge([right_gray, right_gray, right_gray]) / 255.0

            left_pred  = float(eye_model.predict(np.expand_dims(left_img, axis=0), verbose=0)[0][0])
            right_pred = float(eye_model.predict(np.expand_dims(right_img, axis=0), verbose=0)[0][0])
            eye_pred   = (left_pred + right_pred) / 2

            # Yawn
            mouth_img = cv2.resize(mouth, (96,96)) / 255.0
            yawn_pred = float(yawn_model.predict(np.reshape(mouth_img, (1,96,96,3)), verbose=0)[0][0])

            if eye_pred < 0.5:
                eye_text = "CLOSED"
                closed_counter += 1
            else:
                eye_text = "OPEN"
                closed_counter = 0

            if yawn_pred > 0.5:
                yawn_text = "YAWN"
                yawn_counter += 1
            else:
                yawn_text = "NOT YAWN"
                yawn_counter = 0

            no_face_counter = 0
        except Exception as e:
            print(f"[WARN] Crop error: {e}")
    else:
        no_face_counter += 1
        closed_counter = 0
        yawn_counter   = 0

    # ---- Alert Level Logic ----
    drowsiness_score = (closed_counter * 8) + (yawn_counter * 2)

    # 3 States: DISTRACTED, DROWSY, ALERT
    if no_face_counter >= NO_FACE_THRESHOLD:
        alert_level = "DISTRACTED"
        head_text = ""
        eye_text = ""
        yawn_text = ""
    elif distracted_counter >= DISTRACTED_FRAME_THRESHOLD:
        alert_level = "DISTRACTED"
    elif drowsiness_score >= 24:
        alert_level = "DROWSY"
    else:
        alert_level = "ALERT"

    # ---- Record event if alert level changed ----
    now = datetime.now().isoformat(timespec="seconds")
    if alert_level != "ALERT" and alert_level != last_alert:
        last_alert       = alert_level
        last_alert_start = now
        event_log.append({
            "timestamp":  now,
            "event_type": alert_level,
        })

    if alert_level == "ALERT":
        last_alert = None

    # ---- Store history ----
    score_history.append({
        "timestamp":   now,
        "score":       drowsiness_score,
        "alert_level": alert_level,
    })

    return {
        "eye_status":       eye_text,
        "yawn_status":      yawn_text,
        "head_status":      head_text,
        "eye_prob":         round(eye_pred, 3),
        "yawn_prob":        round(yawn_pred, 3),
        "head_prob":        round(head_pred, 3),
        "drowsiness_score": drowsiness_score,
        "distracted_counter": distracted_counter,
        "alert_level":      alert_level,
        "timestamp":        now,
    }


# ============================================================
# GET /history — Last N alertness scores for the live graph
# ============================================================
@app.get("/history")
def get_history():
    return {"history": list(score_history)}


# ============================================================
# GET /events — Full event log for the session
# ============================================================
@app.get("/events")
def get_events():
    return {"events": event_log}


# ============================================================
# GET /reset — Reset all counters (start a new session)
# ============================================================
@app.post("/reset")
def reset_session():
    global closed_counter, yawn_counter, no_face_counter, distracted_counter
    global smooth_yaw, smooth_pitch
    global last_alert, last_alert_start
    closed_counter = yawn_counter = no_face_counter = distracted_counter = 0
    smooth_yaw = smooth_pitch = 0.0
    last_alert = last_alert_start = None
    score_history.clear()
    event_log.clear()
    return {"status": "Session reset successfully."}


# ============================================================
# GET / — Serve the Frontend Dashboard
# ============================================================
@app.get("/")
def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
