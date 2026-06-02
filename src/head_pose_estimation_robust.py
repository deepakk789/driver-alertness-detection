import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# --- UPDATED CONFIGURATION ---
YAW_THRESHOLD = 12    
PITCH_THRESHOLD = 12  
SMOOTHING_FACTOR = 0.1 

# Variables to store smoothed angles
smooth_yaw = 0
smooth_pitch = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    img_h, img_w, img_c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d, face_2d = [], []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 33, 263, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            raw_pitch = angles[0] * 360
            raw_yaw   = angles[1] * 360

            # Smooth the values
            smooth_pitch = (raw_pitch * SMOOTHING_FACTOR) + (smooth_pitch * (1 - SMOOTHING_FACTOR))
            smooth_yaw   = (raw_yaw * SMOOTHING_FACTOR) + (smooth_yaw * (1 - SMOOTHING_FACTOR))

            # Status logic
            status = "FORWARD"
            color = (0, 255, 0)

            if smooth_yaw < -YAW_THRESHOLD: status, color = "LOOKING LEFT", (0, 0, 255)
            elif smooth_yaw > YAW_THRESHOLD: status, color = "LOOKING RIGHT", (0, 0, 255)
            elif smooth_pitch < -PITCH_THRESHOLD: status, color = "LOOKING DOWN", (0, 0, 255)
            elif smooth_pitch > PITCH_THRESHOLD: status, color = "LOOKING UP", (0, 0, 255)

            # Draw
            nose_tip_2d = (int(face_2d[0][0]), int(face_2d[0][1]))
            nose_3d_proj, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            cv2.line(frame, nose_tip_2d, (int(nose_3d_proj[0][0][0]), int(nose_3d_proj[0][0][1])), (255, 255, 0), 2)

            cv2.putText(frame, f"STATUS: {status}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(frame, f"Smooth Yaw: {smooth_yaw:.1f}", (img_w - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Smoothed Head Pose', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
