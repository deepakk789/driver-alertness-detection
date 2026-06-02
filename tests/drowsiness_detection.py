import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from scipy.spatial import distance


data=[]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

EAR_THRESHOLD = 0.20
FRAME_LIMIT = 20

counter = 0

cap = cv2.VideoCapture(0)

def calculate_EAR(eye_points):

    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])

    ear = (A + B) / (2.0 * C)

    return ear


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            for id in LEFT_EYE:
                x = int(face_landmarks.landmark[id].x * w)
                y = int(face_landmarks.landmark[id].y * h)
                left_eye.append((x,y))
                cv2.circle(frame,(x,y),2,(0,255,0),-1)

            for id in RIGHT_EYE:
                x = int(face_landmarks.landmark[id].x * w)
                y = int(face_landmarks.landmark[id].y * h)
                right_eye.append((x,y))
                cv2.circle(frame,(x,y),2,(255,0,0),-1)

            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)

            EAR = (left_EAR + right_EAR) / 2
            label = 0  # 0 = alert, 1 = drowsy

            if EAR < EAR_THRESHOLD:
                counter += 1
                label = 1
            else:
                counter = 0
                label = 0

            # storing data in the datasets
            data.append([EAR, label])

            if counter > FRAME_LIMIT:
                cv2.putText(frame, "DROWSY!", (200,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)

            cv2.putText(frame,f"EAR: {EAR:.2f}",(30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df = pd.DataFrame(data, columns=["EAR", "label"])
df.to_csv("Dataset/ear_dataset.csv", index=False)

print("Dataset saved")

cap.release()
cv2.destroyAllWindows()