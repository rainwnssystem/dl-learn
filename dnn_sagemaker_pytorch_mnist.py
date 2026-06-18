import argparse
import json
import os
import zipfile
import struct

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')


class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.images[idx])

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, int(self.labels[idx])
        return img


class ANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)


def compression(args):
    with zipfile.ZipFile(f'{args.dataset}/mnist.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{args.dataset}/')

    print("====== Dataset loaded ======")


def train(args):
    with open(f'{args.dataset}/dataset/train-images.idx3-ubyte', 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(n, 1, rows, cols)

    with open(f'{args.dataset}/dataset/train-labels.idx1-ubyte', 'rb') as f:
        struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(f'{args.dataset}/dataset/t10k-images.idx3-ubyte', 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        test_images = np.fromfile(f, dtype=np.uint8).reshape(n, 1, rows, cols)

    with open(f'{args.dataset}/dataset/t10k-labels.idx1-ubyte', 'rb') as f:
        struct.unpack('>II', f.read(8))
        test_labels = np.fromfile(f, dtype=np.uint8)

    images      = images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    train_dataset = CustomDataset(images, labels)
    test_dataset  = CustomDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64)

    print("====== Model loaded ======")
    model = ANN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, test_losses         = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(args.epochs):

        # ── Train ──
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        # ── Test ──
        model.eval()
        test_loss, correct_test, total_test = 0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs   = model(images)
                loss      = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted  = torch.max(outputs, 1)
                total_test   += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(100 * correct_test / total_test)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Test Loss: {test_losses[-1]:.4f} | "
              f"Train Acc: {train_accuracies[-1]:.2f}% | "
              f"Test Acc: {test_accuracies[-1]:.2f}%")

    print("Finished Training.")
    save_model(model, args.model_dir)


def save_model(model, model_dir):
    print("Saving the model...")

    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN()
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pth"), map_location=device)
    )
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers",    type=int,   default=2,     metavar="W")
    parser.add_argument("--epochs",     type=int,   default=10,    metavar="E")
    parser.add_argument("--batch_size", type=int,   default=64,    metavar="BS")
    parser.add_argument("--lr",         type=float, default=0.001, metavar="LR")
    parser.add_argument("--hosts",      type=json.loads, default=os.environ["SM_HOSTS"])
    parser.add_argument("--momentum",   type=float, default=0.9,   metavar="M")
    parser.add_argument("--dataset",    type=str,   default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--model-dir",  type=str,   default=os.environ["SM_MODEL_DIR"])

    args = parser.parse_args()
    compression(args)
    train(args)
