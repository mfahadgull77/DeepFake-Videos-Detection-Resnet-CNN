import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your dataset
base_path = "/home/mfahadkhan9@gmail.com/anaconda3/codes/fsgan-master/fsgan/inference/copydeepfake/dataset/videos/fused_dataset_split/"
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduced image size for faster training
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(128),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Only normalization for validation
val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduced image size for faster training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_path, transform=val_transforms)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Define the EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# Define the modified ResNet model with Dropout and BatchNorm
class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Ensure the output is between 0 and 1
        )

    def forward(self, x):
        return self.base_model(x)

model = ModifiedResNet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function, optimizer, and scheduler
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # Smaller learning rate and weight decay
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)  # Reduce LR on a fixed schedule

# Training with early stopping
num_epochs = 50  # Increase the number of epochs
early_stopping = EarlyStopping(patience=7, verbose=True)

train_losses = []
train_accuracies = []
train_precisions = []
train_recalls = []
train_f1s = []

val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        predicted = (outputs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_precision = precision_score(all_labels, all_predictions, zero_division=0)
    train_recall = recall_score(all_labels, all_predictions, zero_division=0)
    train_f1 = f1_score(all_labels, all_predictions, zero_division=0)

    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Training Precision: {train_precision}")
    print(f"Training Recall: {train_recall}")
    print(f"Training F1 Score: {train_f1}")

    # Validation
    model.eval()
    val_running_loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions, zero_division=0)
    val_recall = recall_score(all_labels, all_predictions, zero_division=0)
    val_f1 = f1_score(all_labels, all_predictions, zero_division=0)

    val_accuracies.append(val_accuracy)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)

    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Precision: {val_precision}")
    print(f"Validation Recall: {val_recall}")
    print(f"Validation F1 Score: {val_f1}")

    scheduler.step()
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the best model
model.load_state_dict(torch.load('checkpoint.pt'))

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plotting the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Print average metrics over all epochs
avg_train_loss = sum(train_losses) / len(train_losses)
avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
avg_train_precision = sum(train_precisions) / len(train_precisions)
avg_train_recall = sum(train_recalls) / len(train_recalls)
avg_train_f1 = sum(train_f1s) / len(train_f1s)

avg_val_loss = sum(val_losses) / len(val_losses)
avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
avg_val_precision = sum(val_precisions) / len(val_precisions)
avg_val_recall = sum(val_recalls) / len(val_recalls)
avg_val_f1 = sum(val_f1s) / len(val_f1s)

print(f"Average Training Loss: {avg_train_loss}")
print(f"Average Training Accuracy: {avg_train_accuracy}")
print(f"Average Training Precision: {avg_train_precision}")
print(f"Average Training Recall: {avg_train_recall}")
print(f"Average Training F1 Score: {avg_train_f1}")

print(f"Average Validation Loss: {avg_val_loss}")
print(f"Average Validation Accuracy: {avg_val_accuracy}")
print(f"Average Validation Precision: {avg_val_precision}")
print(f"Average Validation Recall: {avg_val_recall}")
print(f"Average Validation F1 Score: {avg_val_f1}")

# Save the final model
torch.save(model.state_dict(), 'deepfake_detection_model_final.pth')

# Function to evaluate and compute confusion matrix
def evaluate_quality(model, loader, criterion, quality):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {quality}')
    plt.show()

    return avg_loss, accuracy, precision, recall, f1, cm

# Evaluate on the validation dataset
val_loss, val_accuracy, val_precision, val_recall, val_f1, val_cm = evaluate_quality(model, val_loader, criterion, 'Validation')
print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}")

# Save confusion matrix as an image
plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Validation')
plt.savefig('confusion_matrix.png')
