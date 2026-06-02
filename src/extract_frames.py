import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=5):
    """
    Extracts frames from a video and saves them as images.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        frame_interval (int): Save every Nth frame (e.g., 5 means save 1 frame out of every 5).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {video_path}...")

    while True:
        ret, frame = cap.read()
        
        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Save frame based on the interval
        if frame_count % frame_interval == 0:
            # Construct the output filename
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            
            # Save the frame
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Done! Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    # --- Instructions ---
    # 1. Place your recorded videos in the project folder (e.g., "videos/head_left.mp4").
    # 2. Update the video_file and output_folder paths below.
    # 3. Run this script.
    
    # Example usage:
    # Change these paths to match your video and where you want to save the dataset
    video_file = r"../videos/my_head_tilt_video.mp4" 
    output_folder = r"../dataset/head_tilt/left"
    
    # Extract 1 frame every 10 frames (if video is 30fps, this saves 3 frames per second)
    # Adjust frame_interval to get more or fewer frames
    # extract_frames(video_file, output_folder, frame_interval=10)
    
    print("Please edit the script with your video paths and uncomment the extract_frames line to run.")
