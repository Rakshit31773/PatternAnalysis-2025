# predict.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ADNIDataset
from modules import ConvNeXtClassifier

# =========Configuration============
data_root = "/home/groups/comp3710/ADNI/AD_NC/test"  # path to your test dataset
batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/saved_models/convnext_adni.pth"


# =========Data loading============
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_dataset = ADNIDataset(root_dir=data_root, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# =========Model Loading============
model = ConvNeXtClassifier(variant='tiny', number_of_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# =========Prediction and Accuracy============
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")