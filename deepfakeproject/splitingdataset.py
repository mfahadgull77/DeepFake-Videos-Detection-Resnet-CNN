import os
import shutil
from random import shuffle
from tqdm import tqdm

# Define paths to the original genuine and swapped frames
genuine_frames_directory = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/fused_dataset_frames/genuine_frames/"
swapped_frames_directory = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/fused_dataset_frames/swapped_frames/"

# Define paths to the new train and validation directories
base_train_directory = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/fused_dataset_split/train/"
base_val_directory = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/fused_dataset_split/val/"

# Define the percentage of data to use for validation (e.g., 20%)
validation_split = 0.2

# Create train and validation directories if they don't exist
os.makedirs(base_train_directory, exist_ok=True)
os.makedirs(base_val_directory, exist_ok=True)

# Create subdirectories for genuine and swapped classes within train and val directories
train_genuine_directory = os.path.join(base_train_directory, "genuine")
val_genuine_directory = os.path.join(base_val_directory, "genuine")
train_swapped_directory = os.path.join(base_train_directory, "swapped")
val_swapped_directory = os.path.join(base_val_directory, "swapped")

os.makedirs(train_genuine_directory, exist_ok=True)
os.makedirs(val_genuine_directory, exist_ok=True)
os.makedirs(train_swapped_directory, exist_ok=True)
os.makedirs(val_swapped_directory, exist_ok=True)

# Function to split and copy files to the train and validation directories
def split_and_copy_files(source_directory, train_dest, val_dest, validation_split):
    # List all files in the source directory
    files = os.listdir(source_directory)
    shuffle(files)  # Shuffle the files randomly

    # Calculate the number of files for validation
    num_val_files = int(len(files) * validation_split)

    # Split the files into train and validation sets
    train_files = files[num_val_files:]
    val_files = files[:num_val_files]

    # Copy train files to the train destination directory with progress bar
    print(f"Copying train files from {source_directory} to {train_dest}")
    for file in tqdm(train_files, desc="Training Files", unit="file"):
        src_path = os.path.join(source_directory, file)
        dest_path = os.path.join(train_dest, file)
        shutil.copy2(src_path, dest_path)

    # Copy validation files to the validation destination directory with progress bar
    print(f"Copying validation files from {source_directory} to {val_dest}")
    for file in tqdm(val_files, desc="Validation Files", unit="file"):
        src_path = os.path.join(source_directory, file)
        dest_path = os.path.join(val_dest, file)
        shutil.copy2(src_path, dest_path)

# Split and copy genuine frames
split_and_copy_files(genuine_frames_directory, train_genuine_directory, val_genuine_directory, validation_split)

# Split and copy swapped frames
split_and_copy_files(swapped_frames_directory, train_swapped_directory, val_swapped_directory, validation_split)

print("Dataset splitting and classification complete.")
