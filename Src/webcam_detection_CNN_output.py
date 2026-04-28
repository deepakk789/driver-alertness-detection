import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# --- Alarm setup (plays sound when drowsy) ---
try:
    from playsound import playsound
    ALARM_PATH = os.path.join(os.path.dirname(__file__), "..", "Alarm.wav", "alarm.wav")
    ALARM_AVAILABLE = os.path.exists(ALARM_PATH)
except ImportError:
    ALARM_AVAILABLE = False
    print("[INFO] playsound not installed. Alarm will not play. Run: pip install playsound")

# Load models
eye_model = load_model("Models/eye_model.h5")
yawn_model = load_model("Models/yawn_model.h5")
head_model = load_model("Models/head_model_mobilenet_tuned.h5")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indexes (important)
#LEFT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE_FULL = [
    33, 133,   # eye corners
    160, 158, 153, 144,  # eye contour
    70, 63, 105, 66, 107,  # eyebrow
    65, 55, 52,           # upper forehead side
]
#RIGHT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_FULL = [
    362, 263,   # eye corners
    385, 387, 373, 380,  # eye contour
    336, 296, 334, 293, 300,  # eyebrow
    295, 285, 282,       # upper forehead side
]
MOUTH = [
    13, 14, 78, 308,  # main
    82, 87, 317, 312, # upper/lower lips
    95, 88, 178, 87,  # left side
    318, 324, 402, 317 # right side
]

cap = cv2.VideoCapture(0)

# --- Temporal Smoothing Counters ---
# A normal blink lasts ~3-5 frames at 30fps (100-170ms).
# We only flag DROWSY if eyes stay closed for CLOSED_FRAME_THRESHOLD
# consecutive frames, filtering out natural blinks.
CLOSED_FRAME_THRESHOLD = 9   # ~0.4 sec at 30fps → eyes closed this long = drowsy
YAWN_FRAME_THRESHOLD   = 10   # ~0.33 sec at 30fps → sustained yawn = drowsy
NO_FACE_THRESHOLD      = 30   # ~1 sec at 30fps → no face for this long = warning
DISTRACTED_FRAME_THRESHOLD = 25 # ~1 sec at 30fps → looking away for this long = distracted

closed_counter = 0   # counts consecutive "eyes closed" frames
yawn_counter   = 0   # counts consecutive "yawn" frames
no_face_counter = 0  # counts consecutive "no face" frames
distracted_counter = 0 # counts consecutive "looking away" frames
alarm_playing  = False

def crop_region(frame, landmarks, indices, padding=10):
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Force a square bounding box to prevent distortion when resizing to 96x96
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    
    box_w = x_max - x_min
    box_h = y_max - y_min
    max_dim = max(box_w, box_h)
    
    half_size = (max_dim // 2) + padding
    
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + half_size)

    return frame[y1:y2, x1:x2]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Default values
    status = "NO FACE"
    color = (0, 255, 255)
    eye_text = "-"
    yawn_text = "-"
    head_text = "-"
    eye_prob= 0
    yawn_prob = 0
    head_pred = 0
    eye_pred= 0
    yawn_pred = 0
    drowsiness_score = 0
    left_eye = None
    right_eye = None
    mouth = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ==========================================
    # 1. Head Pose Inference (Full Frame)
    # ==========================================
    # The head model was trained on full webcam frames, not cropped faces.
    # Therefore, we pass the full resized frame to it directly.
    # This also allows it to detect "AWAY" even if MediaPipe loses tracking.
    head_img = cv2.resize(rgb, (96, 96)) / 255.0
    head_img = np.reshape(head_img, (1, 96, 96, 3))
    head_pred = head_model.predict(head_img, verbose=0)[0][0]

    # Head Pose prediction (0=AWAY, 1=FORWARD)
    if head_pred < 0.5:
        head_text = "AWAY"
        distracted_counter += 1
    else:
        head_text = "FORWARD"
        distracted_counter = 0

    # ==========================================
    # 2. Eye & Yawn Inference (MediaPipe Crops)
    # ==========================================
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        try:

            # 👁️ Eye crop (using FULL region as trained)
            left_eye = crop_region(frame, landmarks, LEFT_EYE_FULL, padding=10)
            right_eye = crop_region(frame, landmarks, RIGHT_EYE_FULL, padding=10)

            # 😮 Mouth crop
            mouth = crop_region(frame, landmarks, MOUTH ,padding=20)

            # Preprocess eyes: Convert BGR (OpenCV) to RGB (Keras expects RGB)
            left_eye_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
            left_img = cv2.resize(left_eye_rgb, (96, 96)) / 255.0
            left_img = np.reshape(left_img, (1, 96, 96, 3))
            left_pred = eye_model.predict(left_img, verbose=0)[0][0]

            right_eye_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
            right_img = cv2.resize(right_eye_rgb, (96, 96)) / 255.0
            right_img = np.reshape(right_img, (1, 96, 96, 3))
            right_pred = eye_model.predict(right_img, verbose=0)[0][0]

            #Combine results(because model trained on one eye)
            eye_pred = (left_pred + right_pred) / 2

            # Preprocess mouth
            mouth_img = cv2.resize(mouth, (96,96))
            mouth_img = mouth_img / 255.0
            mouth_img = np.reshape(mouth_img, (1,96,96,3))

            yawn_pred = yawn_model.predict(mouth_img, verbose=0)[0][0]

            # 🔥 Decision Logic with Temporal Smoothing
            # Eye prediction
            if eye_pred < 0.5:  # Reverted back to 0.5
                eye_text = "CLOSED"
                closed_counter += 1
            else:
                eye_text = "OPEN"
                closed_counter = 0

            # Yawn prediction
            if yawn_pred > 0.5:
                yawn_text = "YAWN"
                yawn_counter += 1
            else:
                yawn_text = "NOT YAWN"
                yawn_counter = 0

            # Reset no-face counter since we detected a face
            no_face_counter = 0

            # 🔥 Drowsiness Score Logic
            # Eye closure is weighted heavily (x8), yawning is weighted lightly (x2).
            drowsiness_score = (closed_counter * 8) + (yawn_counter * 2)

            if distracted_counter >= DISTRACTED_FRAME_THRESHOLD:
                status = "DISTRACTED"
                color = (0, 0, 255) # Red
                if ALARM_AVAILABLE and not alarm_playing:
                    import threading
                    threading.Thread(target=playsound, args=(ALARM_PATH,), daemon=True).start()
                    alarm_playing = True
            elif drowsiness_score >= 96:
                # E.g., Eyes closed for 12+ frames (12 * 8 = 96)
                status = "HIGH DROWSINESS"
                color = (0, 0, 255) # Red
                if ALARM_AVAILABLE and not alarm_playing:
                    import threading
                    threading.Thread(target=playsound, args=(ALARM_PATH,), daemon=True).start()
                    alarm_playing = True
            elif drowsiness_score >= 40:
                # E.g., Eyes closed for 5+ frames, OR yawning for 20+ frames
                status = "MILD DROWSINESS"
                color = (0, 165, 255) # Orange
                alarm_playing = False
            else:
                status = "ALERT"
                color = (0, 255, 0) # Green
                alarm_playing = False


        except Exception as e:
            print("Error:", e)
    else:
        # No face detected — increment counter
        no_face_counter += 1
        closed_counter = 0
        yawn_counter = 0
        
        # Note: We do NOT reset distracted_counter here. 
        # If no face is detected, they might be looking extremely far away.
        # The head model runs on the full frame and will update distracted_counter.

        if no_face_counter >= NO_FACE_THRESHOLD:
            # If the head model also says they are distracted, show that.
            if distracted_counter >= DISTRACTED_FRAME_THRESHOLD:
                status = "DISTRACTED"
                color = (0, 0, 255)
            else:
                status = "NO FACE"
                color = (0, 255, 255)

    cv2.putText(frame, f"Eye: {eye_text}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Yawn: {yawn_text}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Head: {head_text}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Status: {status}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(frame, f"Score: {drowsiness_score}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Eye Prob: {eye_pred:.2f}", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.putText(frame, f"Yawn Prob: {yawn_pred:.2f}", (300, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.putText(frame, f"Head Prob: {head_pred:.2f}", (300, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Show temporal counters on screen for debugging
    cv2.putText(frame, f"Eye Closed Frames: {closed_counter}/{CLOSED_FRAME_THRESHOLD}", (300, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(frame, f"Yawn Frames: {yawn_counter}/{YAWN_FRAME_THRESHOLD}", (300, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(frame, f"Distracted Frames: {distracted_counter}/{DISTRACTED_FRAME_THRESHOLD}", (300, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        

    # Show main window
    cv2.imshow("Driver Alertness", frame)

    # Show eye windows safely
    if left_eye is not None and left_eye.size > 0:
        cv2.imshow("Left Eye", left_eye)

    if right_eye is not None and right_eye.size > 0:
        cv2.imshow("Right Eye", right_eye)

    if mouth is not None and mouth.size > 0:
        cv2.imshow("Mouth", mouth)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()