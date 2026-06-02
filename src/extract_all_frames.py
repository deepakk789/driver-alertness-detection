import cv2
import os
import glob

def extract_frames(video_path, output_dir, prefix="", frame_interval=10):
    """
    Extracts frames from a video and saves them as images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {os.path.basename(video_path)}...")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"{prefix}frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    base_dir = r"d:\project folder\DRIVER ALLERTNESS DETECTION SYSTEM\HEAD_DATASET\head_tilt"
    output_base_dir = os.path.join(base_dir, "extracted_frames")
    
    # Process "looking_away" (Not Alert)
    looking_away_dir = os.path.join(base_dir, "looking_away")
    out_away = os.path.join(output_base_dir, "looking_away")
    
    for vid_file in glob.glob(os.path.join(looking_away_dir, "*.mp4")):
        # Get video name without extension
        vid_name = os.path.splitext(os.path.basename(vid_file))[0]
        # Prefix frames with video name so they don't overwrite each other
        extract_frames(vid_file, out_away, prefix=f"{vid_name}_", frame_interval=3)

    # Process "looking_forward" (Alert)
    looking_forward_dir = os.path.join(base_dir, "looking_forward")
    out_forward = os.path.join(output_base_dir, "looking_forward")
    
    for vid_file in glob.glob(os.path.join(looking_forward_dir, "*.mp4")):
        vid_name = os.path.splitext(os.path.basename(vid_file))[0]
        extract_frames(vid_file, out_forward, prefix=f"{vid_name}_", frame_interval=3)

    print(f"\nAll frames extracted successfully to: {output_base_dir}")
    print("You can now review them and remove any incorrect frames.")
