import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

mp_face_mesh = mp.solutions.face_mesh

def evaluate_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return
        
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
    
    YAW_THRESHOLD = 12    
    PITCH_THRESHOLD = 12  
    
    y_true = []
    y_pred = []
    
    folder_names = os.listdir(dataset_path)
    forward_dirs = [d for d in folder_names if 'FORWARD' in d.upper()]
    away_dirs = [d for d in folder_names if 'AWAY' in d.upper()]
    
    if not forward_dirs or not away_dirs:
        print(f"Could not find forward/away directories in {dataset_path}")
        return
        
    forward_dir = forward_dirs[0]
    away_dir = away_dirs[0]
    
    class_dirs = {
        0: os.path.join(dataset_path, forward_dir), # 0 is forward
        1: os.path.join(dataset_path, away_dir)     # 1 is away
    }
    
    print(f"\nEvaluating Dataset: {dataset_path}")
    
    total_images = 0
    no_face_detected = 0

    for label, dir_path in class_dirs.items():
        if not os.path.exists(dir_path): continue
        for img_name in os.listdir(dir_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            total_images += 1
            img_path = os.path.join(dir_path, img_name)
            
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            img_h, img_w, img_c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            pred_label = 1 # default to away if face not detected (safe fallback)
            
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
                    
                    if raw_yaw < -YAW_THRESHOLD or raw_yaw > YAW_THRESHOLD or \
                       raw_pitch < -PITCH_THRESHOLD or raw_pitch > PITCH_THRESHOLD:
                        pred_label = 1 # away
                    else:
                        pred_label = 0 # forward
                    break # only process first face
            else:
                no_face_detected += 1
                
            y_true.append(label)
            y_pred.append(pred_label)
            
    print(f"Total Images Evaluated: {total_images}")
    print(f"Images where face mesh failed to detect face: {no_face_detected}")
    if total_images == 0:
        return
        
    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['Forward (0)', 'Away (1)']))
    print("-" * 50)

if __name__ == "__main__":
    cropped_faces_test = r"d:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\DATASET_COMBINED\TEST\HEAD_TEST"
    extracted_frames_test = r"d:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\HEAD_DATASET\head_tilt\extracted_frames"
    
    evaluate_dataset(cropped_faces_test)
    evaluate_dataset(extracted_frames_test)
