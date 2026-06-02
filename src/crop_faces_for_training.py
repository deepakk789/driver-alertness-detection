import cv2
import mediapipe as mp
import os
import glob

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def crop_face(image):
    """
    Detects and crops the face from an image using MediaPipe.
    Returns the cropped face and a success flag.
    """
    h, w, _ = image.shape
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_img)

    if results.detections:
        # Get the first detected face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to pixels
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Add some padding (20%) to capture the whole head
        padding_w = int(width * 0.2)
        padding_h = int(height * 0.2)

        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(w, x + width + padding_w)
        y2 = min(h, y + height + padding_h)

        return image[y1:y2, x1:x2], True
    
    return None, False

def process_dataset(input_base, output_base):
    """
    Processes all images in looking_away and looking_forward subfolders.
    """
    categories = ["looking_away", "looking_forward"]
    
    for category in categories:
        input_dir = os.path.join(input_base, category)
        output_dir = os.path.join(output_base, category)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created: {output_dir}")

        image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
        print(f"Processing {len(image_files)} images in {category}...")

        count = 0
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None: continue

            face, success = crop_face(img)
            if success:
                # Resize to a standard size for the model (e.g., 96x96)
                face_resized = cv2.resize(face, (96, 96))
                
                out_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(output_dir, out_name), face_resized)
                count += 1
        
        print(f"Finished {category}: Cropped {count} faces.")

if __name__ == "__main__":
    # Update these paths to match your folder structure
    INPUT_FOLDER = r"d:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\HEAD_DATASET\head_tilt\extracted_frames"
    OUTPUT_FOLDER = r"d:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\HEAD_DATASET\head_tilt\cropped_faces"

    if os.path.exists(INPUT_FOLDER):
        process_dataset(INPUT_FOLDER, OUTPUT_FOLDER)
        print(f"\nSuccess! All cropped faces are in: {OUTPUT_FOLDER}")
    else:
        print(f"Error: Could not find input folder at {INPUT_FOLDER}")
