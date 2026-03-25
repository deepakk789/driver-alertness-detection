import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load models
eye_model = load_model("Models/eye_model.h5")
yawn_model = load_model("Models/yawn_model.h5")

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

def crop_region(frame, landmarks, indices, padding=10):
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min = max(0, min(x_coords) - padding)
    x_max = min(w, max(x_coords) + padding)
    y_min = max(0, min(y_coords) - padding)
    y_max = min(h, max(y_coords) + padding)

    return frame[y_min:y_max, x_min:x_max]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Default values (always shown)
    status = "NO FACE"
    color = (0, 255, 255)
    eye_text = "-"
    yawn_text = "-"
    eye_prob= 0
    yawn_prob = 0
    eye_pred= 0
    yawn_pred = 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        try:

            # 👁️ Eye crop
            left_eye = crop_region(frame, landmarks, LEFT_EYE_FULL)
            right_eye = crop_region(frame, landmarks, RIGHT_EYE_FULL)

            # 😮 Mouth crop
            mouth = crop_region(frame, landmarks, MOUTH ,padding=20)

            # Preprocess eyes
            # Left eye
            left_img = cv2.resize(left_eye, (96, 96)) / 255.0
            left_img = np.reshape(left_img, (1, 96, 96, 3))
            left_pred = eye_model.predict(left_img, verbose=0)[0][0]

            # Right eye
            right_img = cv2.resize(right_eye, (96, 96)) / 255.0
            right_img = np.reshape(right_img, (1, 96, 96, 3))
            right_pred = eye_model.predict(right_img, verbose=0)[0][0]

            #Combine results(because model trained on one eye)
            eye_pred = (left_pred + right_pred) / 2

            # Preprocess mouth
            mouth_img = cv2.resize(mouth, (96,96))
            mouth_img = mouth_img / 255.0
            mouth_img = np.reshape(mouth_img, (1,96,96,3))

            yawn_pred = yawn_model.predict(mouth_img, verbose=0)[0][0]

            # 🔥 Decision Logic
            # Eye prediction
            if eye_pred < 0.5:
                eye_text = "CLOSED"
            else:
                eye_text = "OPEN"

            # Yawn prediction
            if yawn_pred > 0.5:
                yawn_text = "YAWN"
            else:
                yawn_text = "NOT YAWN"

            if eye_text == "CLOSED" or yawn_text == "YAWN":
                status = "DROWSY"
                color = (0,0,255)
            else:
                status = "ALERT"
                color = (0,255,0)


        except Exception as e:
            print("Error:", e)

    cv2.putText(frame, f"Eye: {eye_text}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Yawn: {yawn_text}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Status: {status}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(frame, f"Eye Prob: {eye_pred:.2f}", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.putText(frame, f"Yawn Prob: {yawn_pred:.2f}", (300, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        

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