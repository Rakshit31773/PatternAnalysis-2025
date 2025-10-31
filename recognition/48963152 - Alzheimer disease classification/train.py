import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from dataset import ADNIDataset
from modules import ConvNeXtClassifier



# =========Configuration============
model_path = "/saved_models/convnext_adni.pth"
data_root = "/home/groups/comp3710/ADNI/AD_NC/train" #root directory of the training data
batch_size = 32    #Number of samples processed before the model is updated
num_epochs = 30       #Number of times the Training loop will iterate over the whole dataset
learning_rate = 1e-4 #controls how much to change the model according to the error
val_split = 0.2       #The portion of training dataset that will be used for validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========Data Loading============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),                     #ConvNeXt expects images of size 224x224
    transforms.ToTensor(),                             #Scales the pixel values to 0 and 1
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = ADNIDataset(root_dir=data_root, transform=train_transform)

val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# =========Model Initialization============
model = ConvNeXtClassifier(variant='tiny', number_of_classes=2, dropout=0.3)
model = model.to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# =========Traversing through the dataset============
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    scheduler.step()

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


# =========Saving the model============
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/convnext_adni.pth")
print("Model saved!")

# =========Plotting the loss and accuracy graphs============
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.savefig("plots/training_plot.png")
plt.close()
