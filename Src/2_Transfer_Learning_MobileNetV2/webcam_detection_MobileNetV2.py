"""
Driver Alertness Detection — Real-Time Webcam Inference
========================================================
Model: MobileNetV2 (Transfer Learning)
  - eye_model_mobilenet_tuned.h5
  - yawn_model_mobilenet_tuned.h5
  - head_model_mobilenet_tuned.h5

IMPROVEMENTS:
  - Added Grayscale-to-RGB conversion to match MRL dataset training domain.
  - Tighter eye crop (removed eyebrow/forehead landmarks) to focus on the eye.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# --- Alarm setup ---
try:
    from playsound import playsound
    # Path is relative to Src/2_Transfer_Learning_MobileNetV2/
    ALARM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Alarm.wav", "alarm.wav")
    ALARM_AVAILABLE = os.path.exists(ALARM_PATH)
except ImportError:
    ALARM_AVAILABLE = False
    print("[INFO] playsound not installed.")

# Load models
eye_model  = load_model("Models/eye_model_mobilenet_tuned.h5")
yawn_model = load_model("Models/yawn_model_mobilenet_tuned.h5")
head_model = load_model("Models/head_model_mobilenet_tuned.h5")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Tighter Eye Landmarks (Standard Contour)
LEFT_EYE  = [33, 133, 160, 158, 153, 144]
RIGHT_EYE = [362, 263, 385, 387, 373, 380]
MOUTH     = [13, 14, 78, 308, 82, 87, 317, 312]

cap = cv2.VideoCapture(0)

# Temporal Smoothing Counters
CLOSED_FRAME_THRESHOLD = 9
YAWN_FRAME_THRESHOLD   = 10
NO_FACE_THRESHOLD      = 30
DISTRACTED_FRAME_THRESHOLD = 25

closed_counter = 0
yawn_counter   = 0
no_face_counter = 0
distracted_counter = 0
alarm_playing  = False

def crop_region(frame, landmarks, indices, padding=10):
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    max_dim = max(x_max - x_min, y_max - y_min)
    half_size = (max_dim // 2) + padding
    
    x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
    x2, y2 = min(w, cx + half_size), min(h, cy + half_size)
    
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def preprocess_eye(eye_crop):
    """
    Matches MRL Dataset Domain:
    1. Grayscale conversion
    2. Resize to 96x96
    3. Back to 3-channel (since model expects RGB input)
    4. Normalize /255.0
    """
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96))
    rgb_like = cv2.merge([resized, resized, resized])
    return np.expand_dims(rgb_like / 255.0, axis=0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Display elements
    status = "NO FACE"
    color = (0, 255, 255)
    eye_text, yawn_text, head_text = "-", "-", "-"
    eye_pred, yawn_pred, head_pred = 0, 0, 0
    drowsiness_score = 0
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Head Pose (Full Frame)
    head_img = cv2.resize(rgb, (96, 96)) / 255.0
    head_img = np.reshape(head_img, (1, 96, 96, 3))
    head_pred = head_model.predict(head_img, verbose=0)[0][0]

    if head_pred < 0.5:
        head_text = "AWAY"
        distracted_counter += 1
    else:
        head_text = "FORWARD"
        distracted_counter = 0

    # 2. Eye & Yawn (Crops)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        try:
            # Tighter eye crops (padding=5 for focus)
            left_eye, l_box  = crop_region(frame, landmarks, LEFT_EYE, padding=5)
            right_eye, r_box = crop_region(frame, landmarks, RIGHT_EYE, padding=5)
            mouth, m_box     = crop_region(frame, landmarks, MOUTH, padding=4)

            # Draw Green Boxes on the main frame
            for (x1, y1, x2, y2) in [l_box, r_box, m_box]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Preprocess using Grayscale Fix
            left_input  = preprocess_eye(left_eye)
            right_input = preprocess_eye(right_eye)
            
            left_prob  = eye_model.predict(left_input, verbose=0)[0][0]
            right_prob = eye_model.predict(right_input, verbose=0)[0][0]
            eye_pred   = (left_prob + right_prob) / 2

            # Mouth Preprocess (Standard RGB)
            mouth_rgb = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
            mouth_input = np.expand_dims(cv2.resize(mouth_rgb, (96, 96)) / 255.0, axis=0)
            yawn_pred = yawn_model.predict(mouth_input, verbose=0)[0][0]

            # Logic
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
            drowsiness_score = (closed_counter * 8) + (yawn_counter * 2)

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
            print("Error:", e)
    else:
        no_face_counter += 1
        closed_counter, yawn_counter = 0, 0
        if no_face_counter >= NO_FACE_THRESHOLD:
            status = "DISTRACTED" if distracted_counter >= DISTRACTED_FRAME_THRESHOLD else "NO FACE"
            color = (0, 0, 255) if status == "DISTRACTED" else (0, 255, 255)

    # UI - Main Info
    cv2.putText(frame, f"Status: {status}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.putText(frame, f"Score: {drowsiness_score}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # UI - Individual Labels
    cv2.putText(frame, f"Eye: {eye_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Yawn: {yawn_text}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Head: {head_text}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # UI - Probabilities
    cv2.putText(frame, f"Eye Prob: {eye_pred:.2f}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Yawn Prob: {yawn_pred:.2f}", (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Head Prob: {head_pred:.2f}", (300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # UI - Temporal Counters
    cv2.putText(frame, f"Closed: {closed_counter}/{CLOSED_FRAME_THRESHOLD}", (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(frame, f"Yawn: {yawn_counter}/{YAWN_FRAME_THRESHOLD}", (300, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(frame, f"Distracted: {distracted_counter}/{DISTRACTED_FRAME_THRESHOLD}", (300, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    cv2.putText(frame, "Model: MobileNetV2 (G-Fix)", (30, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    cv2.imshow("Driver Alertness [MobileNetV2]", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()