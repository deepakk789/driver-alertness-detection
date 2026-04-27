# Driver Alertness Detection System

This is a real-time system to detect if a driver is getting sleepy or losing focus. It uses a webcam to track eye closure and yawning patterns, triggering an alarm if it detects signs of drowsiness over several frames.

The system is built using Python, OpenCV, and MediaPipe for facial tracking, with custom-trained CNN models to classify the eye and mouth states.

## How it works

The detection pipeline follows a few simple steps:
1. It grabs frames from the webcam and uses MediaPipe's Face Mesh to find 468 facial landmarks.
2. It crops out the eye and mouth regions based on these landmarks.
3. These crops are fed into two separate CNN models (one for eyes, one for yawning).
4. Instead of just looking at a single frame, it uses a "temporal smoothing" approach. This means it counts how many consecutive frames your eyes are closed or you're yawning. This helps ignore quick blinks or small mouth movements.
5. If your eyes are closed for more than 12 frames or you're yawning for more than 10 frames, the status changes to "DROWSY" and an alarm plays.

## Project Structure

- **Src/**: Contains the training scripts and the main detection script (`webcam_detection_CNN_output.py`).
- **Models/**: Pre-trained weights for the CNN models and MobileNetV2 comparisons.
- **DATASET_COMBINED/**: The images used for training (roughly 60,000 images total).
- **Alarm.wav/**: The audio files for the alert system.

## The Models

I used a custom 3-layer CNN for the main detection because it's lightweight and fast for real-time use. I also included some scripts to compare this against MobileNetV2 (both frozen and fine-tuned) to see how transfer learning performs on this specific task.

The custom models were trained on:
- **Eye Dataset**: ~54,000 images of open and closed eyes.
- **Yawn Dataset**: ~6,000 images of yawning and non-yawning mouths.

## Getting Started

### 1. Setup
Clone the repo and set up a virtual environment:
```bash
git clone https://github.com/yourusername/driver-alertness-detection.git
cd driver-alertness-detection
python -m venv venv
# Activate it (Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run it
Just run the main script to start the webcam detection:
```bash
python Src/webcam_detection_CNN_output.py
```

### 4. Comparison (Optional)
If you want to see how the MobileNetV2 models compare to the custom CNN:
```bash
python Src/train_eye_model_mobilenetv2.py
python Src/train_yawn_model_mobilenetv2.py
```

## Controls
- Press **Esc** to close the window and stop the program.

---
*Note: This project was made for educational purposes to explore real-time computer vision and deep learning.*
