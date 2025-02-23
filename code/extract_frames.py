import cv2
import os

def extract_frames(video_path, output_folder, frames_per_video=10):
    """Extract frames from a video and save them as images."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_per_video):
        frame_num = int((i / frames_per_video) * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if success:
            frame_filename = os.path.join(output_folder, f"{os.path.basename(video_path)}_frame_{i}.jpg")
            cv2.imwrite(frame_filename, frame)

    cap.release()

def process_dataset(video_root, output_root, frames_per_video=10):
    """Extract frames from 'real' and 'fake' videos and save them in labeled folders."""
    for label in ["real", "fake"]:  # Process both classes
        video_dir = os.path.join(video_root, label)
        output_dir = os.path.join(output_root, label)
        
        os.makedirs(output_dir, exist_ok=True)

        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            extract_frames(video_path, output_dir, frames_per_video)

# Run for both training and test datasets
process_dataset("data/videos_train", "data/train")
process_dataset("data/videos_test", "data/test")
