import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_DIR = "data/frames"
OUTPUT_MODEL = "models/emergency_classifier_resnet18.pt"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path("models").mkdir(exist_ok=True)

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
    class_names = dataset.classes
    print("Classes:", class_names)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # important for Windows
    )

    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_feats = base_model.fc.in_features
    base_model.fc = nn.Linear(num_feats, len(class_names))
    model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    save_obj = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(save_obj, OUTPUT_MODEL)
    print("Saved model to", OUTPUT_MODEL)


if __name__ == "__main__":
    main()
