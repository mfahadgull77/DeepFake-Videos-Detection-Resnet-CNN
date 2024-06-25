import cv2
import os
from tqdm import tqdm

# Paths
genuine_video_path = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/genuine_videos/"
swapped_video_path = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/Swapped_videos/"
frame_save_path = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/fused_dataset_frames/"

def extract_frames_from_video(video_path, save_path, label):
    # Create separate folders for genuine and swapped frames
    if label == 'genuine':
        save_path = os.path.join(save_path, 'genuine_frames')
    elif label == 'swapped':
        save_path = os.path.join(save_path, 'swapped_frames')
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(save_path, f"{label}_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

# Extract frames from genuine videos
genuine_videos = os.listdir(genuine_video_path)
for video_file in tqdm(genuine_videos, desc="Processing Genuine Videos"):
    extract_frames_from_video(os.path.join(genuine_video_path, video_file), frame_save_path, 'genuine')

# Extract frames from swapped videos
swapped_videos = os.listdir(swapped_video_path)
for video_file in tqdm(swapped_videos, desc="Processing Swapped Videos"):
    extract_frames_from_video(os.path.join(swapped_video_path, video_file), frame_save_path, 'swapped')

print("All videos have been processed and frames have been extracted.")
