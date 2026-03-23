import cv2
import mediapipe as mp
import os
import pandas as pd
from scipy.spatial import distance

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

data = []

def calculate_EAR(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def process_folder(folder_path, label):
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = img.shape
                left_eye = []
                right_eye = []

                for id in LEFT_EYE:
                    x = int(face_landmarks.landmark[id].x * w)
                    y = int(face_landmarks.landmark[id].y * h)
                    left_eye.append((x,y))

                for id in RIGHT_EYE:
                    x = int(face_landmarks.landmark[id].x * w)
                    y = int(face_landmarks.landmark[id].y * h)
                    right_eye.append((x,y))

                EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2
                data.append([EAR, label])

# process dataset
process_folder("CEW/openEyes", 0)
process_folder("CEW/closedEyes", 1)

# save CSV
df = pd.DataFrame(data, columns=["EAR", "label"])
df.to_csv("Dataset/ear_dataset.csv", index=False)

print("Dataset created from CEW ")