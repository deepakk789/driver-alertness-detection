import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Landmark indexes
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

# --- TUNING PARAMETERS ---
# Change these values to test what the crop looks like!
# Negative padding (e.g. -5) makes the crop even tighter.
EYE_PADDING = 10 
MOUTH_PADDING = 4
# -------------------------

cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    display_frame = frame.copy()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        try:
            # Crop regions
            left_eye = crop_region(frame, landmarks, LEFT_EYE_FULL, padding=EYE_PADDING)
            right_eye = crop_region(frame, landmarks, RIGHT_EYE_FULL, padding=EYE_PADDING)
            mouth = crop_region(frame, landmarks, MOUTH, padding=MOUTH_PADDING)

            # Display the crops in separate small windows
            if left_eye.size > 0:
                cv2.imshow("Left Eye Crop", left_eye)
            if right_eye.size > 0:
                cv2.imshow("Right Eye Crop", right_eye)
            if mouth.size > 0:
                cv2.imshow("Mouth Crop", mouth)
                
            cv2.putText(display_frame, f"Eye Padding: {EYE_PADDING}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mouth Padding: {MOUTH_PADDING}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Modify padding vars in script to test", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        except Exception as e:
            pass

    cv2.imshow("Original Webcam", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
