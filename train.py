import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import ImageFolder
from model import SimpleCNN as base_cnn
# from model.pth import STH


def train_model(data_path="dataset", batch_size=16, EPOCH=5):
    # SET:

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((1.0, 1.0, 1.0), (1.5, 1.5, 1.5))
    ])

    dataset = ImageFolder(data_path, transform=transform)
    data_loader = (DataLoader(dataset, batch_size=batch_size, shuffle=True))

    classes = dataset.classes
    number_classes = len(classes)
    # print(f"Classes: {classes}")

    ##############

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base_cnn(number_classes).to(my_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start Training...")

    for epoch in range(EPOCH):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(my_device), labels.to(my_device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCH}, Loss: {running_loss/len(data_loader):.4f}")

    print("✅ Training Finished!")
###########

    torch.save(model.state_dict(), "model.pth")
    print("💾 Model saved as model.pth")
    return model


if __name__ == "__main__":
    train_model()
