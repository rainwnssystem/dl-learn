import os
import struct
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



# ─────────────────────────────────────────────
#  CustomDataset
# ─────────────────────────────────────────────

class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # if it already translated to numpy array or tensor
        img = self.images[idx]
        img = torch.tensor(img)

        # else (path list)
        # img = Image.open(self.images[idx]).convert('L')  # grayscale='L', RGB='RGB'
        # img = transforms.ToTensor()(img)

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            return img, int(self.labels[idx])
        return img



# ─────────────────────────────────────────────
#  1. idx-ubyte
# ─────────────────────────────────────────────

transform = transforms.Normalize((0.5,), (0.5,))

with open('train-images.idx3-ubyte', 'rb') as f:
    _, n, rows, cols = struct.unpack('>IIII', f.read(16))
    train_images = np.fromfile(f, dtype=np.uint8).reshape(n, 1, rows, cols)

with open('train-labels.idx1-ubyte', 'rb') as f:
    struct.unpack('>II', f.read(8))
    train_labels = np.fromfile(f, dtype=np.uint8)

with open('t10k-images.idx3-ubyte', 'rb') as f:
    _, n, rows, cols = struct.unpack('>IIII', f.read(16))
    test_images = np.fromfile(f, dtype=np.uint8).reshape(n, 1, rows, cols)

with open('t10k-labels.idx1-ubyte', 'rb') as f:
    struct.unpack('>II', f.read(8))
    test_labels = np.fromfile(f, dtype=np.uint8)

train_images = train_images.astype(np.float32) / 255.0
test_images  = test_images.astype(np.float32)  / 255.0

train_dataset = CustomDataset(train_images, train_labels, transform)
test_dataset  = CustomDataset(test_images,  test_labels,  transform)



# ─────────────────────────────────────────────
#  2. pkl
# ─────────────────────────────────────────────

transform = transforms.Normalize((0.5,), (0.5,))

with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('test.pkl', 'rb') as f:
    test_data = pickle.load(f)

train_data['images'] = train_data['images'].astype(np.float32) / 255.0
test_data['images']  = test_data['images'].astype(np.float32)  / 255.0

# print(type(train_data)) / print(train_data.keys()) 로 구조 확인 후 아래 수정
train_dataset = CustomDataset(train_data['images'], train_data['labels'], transform)
test_dataset  = CustomDataset(test_data['images'],  test_data['labels'],  transform)



# ─────────────────────────────────────────────
#  3. png
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_paths_labels(root):
    paths, labels = [], []
    for label, cls in enumerate(sorted(os.listdir(root))):
        cls_dir = os.path.join(root, cls)
        for fname in os.listdir(cls_dir):
            paths.append(os.path.join(cls_dir, fname))
            labels.append(label)
    return paths, labels

train_paths, train_labels = get_paths_labels('./data/train')
test_paths,  test_labels  = get_paths_labels('./data/test')

# __getitem__ 에서 경로 처리 코드(주석)로 교체
train_dataset = CustomDataset(train_paths, train_labels, transform)
test_dataset  = CustomDataset(test_paths,  test_labels,  transform)



# ─────────────────────────────────────────────
#  DataLoader
# ─────────────────────────────────────────────

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64)