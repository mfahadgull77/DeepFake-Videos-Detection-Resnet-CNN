
# Deepfake Detection using Modified ResNet

This project involves detecting deepfake videos using a modified ResNet model with additional convolutional layers pout and batch normalization. The code is designed to handle data loading, model training, evaluation, and visualization of results, including confusion matrices.

## Requirements

To run this project, you'll need to have the following libraries installed:

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn

You can install the required libraries using pip:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

Alternatively, you can create a conda environment with all dependencies:

```bash
conda create -n deepfake_detection python=3.7
conda activate DeepFake
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install scikit-learn matplotlib seaborn
```
The dataset can be downloaded from the links provided below.
1. https://github.com/EndlessSora/DeeperForensics-1.0
2. https://github.com/ondyari/FaceForensics/blob/master/dataset/
3. https://github.com/yuezunli/celeb-deepfakeforensics
4. We have created our own private dataset that is not available as a public dataset.

Note: All these datasets are merged into one fused dataset for using in this project.

## Dataset Preparation

### Step 1:Convert Videos into Frames
You need to convert your videos into frames.

### Step 2: Dataset Structure

Organize your dataset in the following structure:

```
dataset/
    videos/
	Frames/
        fused_dataset_split/
            train/
                class1/
                class2/
            val/
                class1/
                class2/
```
```

### Step 3: Split Dataset into Train and Validation Sets

Ensure that your dataset is already split into training and validation sets with the structure mentioned in Step 2.You can manually move frames into the respective folders.

## Training the Model

Ensure that the paths to your dataset are correctly set:


### Evaluation

The script includes evaluation of the model on the validation dataset after each epoch. It uses various metrics such as accuracy, precision, recall, and F1-score.

### Saving and Loading the Model

The best model is saved as `checkpoint.pt`. You can load this model for further evaluation or inference.

## Results Visualization

The script includes plotting of training and validation loss, accuracy, and the confusion matrix.

### Loss and Accuracy Plots

After training, the script generates plots for training and validation loss and accuracy.

### Confusion Matrix

The confusion matrix for the validation dataset is displayed and saved as `confusion_matrix.png`.

## Conclusion

This project demonstrates a complete workflow for training and evaluating a deepfake detection model using a modified ResNet with multiple convolutional layers and maxpooling.The provided details includes data loading, model training, evaluation, and visualization of results.

Feel free to modify the code according to your dataset and requirements.

## Author

This project is developed by [Muhammad Fahad]

For any queries or issues, please contact [fahadgull77@tju.edu.cn]
