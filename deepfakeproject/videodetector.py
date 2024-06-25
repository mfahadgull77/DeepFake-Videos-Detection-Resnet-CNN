import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load your trained deepfake detection model
model = keras.models.load_model("deepfake_detector_model.h5")

# Function to capture frames from a video
def capture_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

# Function to preprocess a list of frames
def preprocess_frames(frames, target_size=(160, 160)):
    preprocessed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size)
        normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
        preprocessed_frames.append(normalized_frame)
    return np.array(preprocessed_frames)

# Function to predict if a video is real or fake
def predict_video(video_path):
    frames = capture_frames(video_path)
    preprocessed_frames = preprocess_frames(frames)
    predictions = model.predict(preprocessed_frames)
    
    # Aggregate the predictions (e.g., take the mean or majority vote)
    mean_prediction = np.mean(predictions)
    return mean_prediction

# Example usage
video_path = "/home/Fahad/anaconda3/codes/fsgan-master/fsgan/inference/dataset/Swapped_videos/yasir_fahad_new.mp4"
prediction = predict_video(video_path)

# Set a threshold to determine real or fake
threshold = 0.8
if prediction >= threshold:
    print("Fake Video")
else:
    print("Real Video")
