"""
Driver Alertness Detection — Real-Time Webcam Inference
========================================================
Model: Custom CNN (trained from scratch)
  - eye_model.h5   → Eye open/closed classification
  - yawn_model.h5  → Yawn/no-yawn classification
  - head_model.h5  → Head pose (FORWARD / AWAY) classification

Run this script from the project root directory:
    python Src/1_Custom_CNN/webcam_detection_CustomCNN.py
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# --- Alarm setup (plays sound when drowsy) ---
try:
    from playsound import playsound
    ALARM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Alarm.wav", "alarm.wav")
    ALARM_AVAILABLE = os.path.exists(ALARM_PATH)
except ImportError:
    ALARM_AVAILABLE = False
    print("[INFO] playsound not installed. Alarm will not play. Run: pip install playsound")

# ===========================
# Load Custom CNN Models
# ===========================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
eye_model  = load_model(os.path.join(BASE_DIR, "Models", "eye_model.h5"))
yawn_model = load_model(os.path.join(BASE_DIR, "Models", "yawn_model.h5"))
head_model = load_model(os.path.join(BASE_DIR, "Models", "head_model.h5"))

print("[INFO] Custom CNN models loaded successfully.")

# ===========================
# MediaPipe Face Mesh Setup
# ===========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Tighter Landmark Indices (Focused on features)
LEFT_EYE = [33, 133, 160, 158, 153, 144]   # eye contour only
RIGHT_EYE = [362, 263, 385, 387, 373, 380] # eye contour only
MOUTH = [61, 291, 0, 17] # Extreme outer edges of lips

cap = cv2.VideoCapture(0)

# ===========================
# Temporal Smoothing Thresholds
# ===========================
# A normal blink lasts ~3-5 frames at 30fps (100-170ms).
# We only flag DROWSY if eyes stay closed past these thresholds.
CLOSED_FRAME_THRESHOLD    = 9    # ~0.4 sec at 30fps → eyes closed this long = drowsy
YAWN_FRAME_THRESHOLD      = 10   # ~0.33 sec at 30fps → sustained yawn = drowsy
NO_FACE_THRESHOLD         = 30   # ~1 sec at 30fps → no face for this long = warning
DISTRACTED_FRAME_THRESHOLD = 25  # ~1 sec at 30fps → looking away for this long = distracted

closed_counter    = 0
yawn_counter      = 0
no_face_counter   = 0
distracted_counter = 0
alarm_playing     = False


def crop_region(frame, landmarks, indices, padding=10):
    """Crop a square bounding box around the given landmark indices and return coordinates."""
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Force a square bounding box to prevent distortion when resizing to 96x96
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    max_dim = max(x_max - x_min, y_max - y_min)
    half_size = (max_dim // 2) + padding

    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + half_size)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ===========================
# Main Detection Loop
# ===========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Default display values
    status         = "DISTRACTED"
    color          = (0, 255, 255)
    eye_text       = "-"
    yawn_text      = "-"
    head_text      = "-"
    eye_pred       = 0.0
    yawn_pred      = 0.0
    head_pred      = 0.0
    drowsiness_score = 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------
    # 1. Head Pose Inference (Full Frame)
    # ------------------------------------------
    head_img = cv2.resize(rgb, (96, 96)) / 255.0
    head_img = np.reshape(head_img, (1, 96, 96, 3))
    head_pred = head_model.predict(head_img, verbose=0)[0][0]

    if head_pred < 0.5:
        head_text = "AWAY"
        distracted_counter += 1
    else:
        head_text = "FORWARD"
        distracted_counter = 0

    # ------------------------------------------
    # 2. Eye & Yawn Inference (MediaPipe Crops)
    # ------------------------------------------
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        try:
            # Crop eye and mouth regions (Reduced padding to 5 for tight fit)
            left_eye, l_box  = crop_region(frame, landmarks, LEFT_EYE, padding=5)
            right_eye, r_box = crop_region(frame, landmarks, RIGHT_EYE, padding=5)
            mouth, m_box     = crop_region(frame, landmarks, MOUTH, padding=0)

            # Draw Green Boxes on the main frame
            for (x1, y1, x2, y2) in [l_box, r_box, m_box]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Preprocess left eye: BGR→RGB, resize, normalize
            left_eye_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
            left_img     = cv2.resize(left_eye_rgb, (96, 96)) / 255.0
            left_img     = np.reshape(left_img, (1, 96, 96, 3))
            left_pred    = eye_model.predict(left_img, verbose=0)[0][0]

            # Preprocess right eye
            right_eye_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
            right_img     = cv2.resize(right_eye_rgb, (96, 96)) / 255.0
            right_img     = np.reshape(right_img, (1, 96, 96, 3))
            right_pred    = eye_model.predict(right_img, verbose=0)[0][0]

            eye_pred = (left_pred + right_pred) / 2

            # Preprocess mouth
            mouth_rgb = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
            mouth_img = cv2.resize(mouth_rgb, (96, 96)) / 255.0
            mouth_img = np.reshape(mouth_img, (1, 96, 96, 3))
            yawn_pred = yawn_model.predict(mouth_img, verbose=0)[0][0]

            # ---- Eye Decision ----
            if eye_pred < 0.5:
                eye_text = "CLOSED"
                closed_counter += 1
            else:
                eye_text = "OPEN"
                closed_counter = 0

            # ---- Yawn Decision ----
            if yawn_pred > 0.5:
                yawn_text = "YAWN"
                yawn_counter += 1
            else:
                yawn_text = "NOT YAWN"
                yawn_counter = 0

            no_face_counter = 0
            drowsiness_score = (closed_counter * 8) + (yawn_counter * 2)

            # ---- Status Logic ----
            if distracted_counter >= DISTRACTED_FRAME_THRESHOLD:
                status, color = "DISTRACTED", (0, 0, 255)
            elif drowsiness_score >= 96:
                status, color = "HIGH DROWSINESS", (0, 0, 255)
            elif drowsiness_score >= 40:
                status, color = "MILD DROWSINESS", (0, 165, 255)
            else:
                status, color = "ALERT", (0, 255, 0)

            # Alarm
            if (status in ["DISTRACTED", "HIGH DROWSINESS"]) and ALARM_AVAILABLE and not alarm_playing:
                import threading
                threading.Thread(target=playsound, args=(ALARM_PATH,), daemon=True).start()
                alarm_playing = True
            elif status not in ["DISTRACTED", "HIGH DROWSINESS"]:
                alarm_playing = False

        except Exception as e:
            print("Error during inference:", e)

    else:
        no_face_counter  += 1
        closed_counter    = 0
        yawn_counter      = 0
        if no_face_counter >= NO_FACE_THRESHOLD:
            status = "DISTRACTED"
            color = (0, 0, 255)

    # ===========================
    # Display Overlay
    # ===========================
    cv2.putText(frame, f"Status: {status}",  (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,   color, 3)
    cv2.putText(frame, f"Score: {drowsiness_score}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Eye: {eye_text}",   (30, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Yawn: {yawn_text}", (30, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Head: {head_text}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Eye Prob:  {eye_pred:.2f}",  (300, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Yawn Prob: {yawn_pred:.2f}", (300, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Head Prob: {head_pred:.2f}", (300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Closed: {closed_counter}/{CLOSED_FRAME_THRESHOLD}",
                (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"Yawn: {yawn_counter}/{YAWN_FRAME_THRESHOLD}",
                (300, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, f"Distracted: {distracted_counter}/{DISTRACTED_FRAME_THRESHOLD}",
                (300, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(frame, "Model: Custom CNN", (30, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Driver Alertness [Custom CNN]", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

