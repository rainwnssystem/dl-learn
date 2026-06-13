import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import copy


train_transform = transforms.Compose([
    # Data Augmentation: 학습 데이터를 인위적으로 변경하여 다양성을 높인다.
    transforms.Resize((128, 128)),

    # 상하좌우 32px zero-padding 후 256×256 랜덤 크롭, 물체 위치가 조금씩 달라진다.
    transforms.RandomCrop(128, padding=16),

    # 50% 확률로 좌우 반전
    transforms.RandomHorizontalFlip(),
    
    # 밝기 및 대비 ±20% 랜덤 변화
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    # PIL Image, Numpy array -> PyTorch Tensor
    transforms.ToTensor(),
    
    # 채널별 정규화: (pixel - mean) / std
    transforms.Normalize(MEAN, STD),
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])



# ImageFolder로 train, test 폴더 내부 class 폴더 자동 인식
dataset = datasets.ImageFolder(root='./datasets/...', transform=None)

# Dataset에서 일부만 사용
indices = torch.randperm(len(dataset))[:5000]
small_dataset = Subset(dataset, indices)


train_dataset, test_dataset = random_split(small_dataset, [0.8, 0.2])  # train, test 분리

train_dataset.dataset = copy.copy(dataset)
test_dataset.dataset  = copy.copy(dataset)

train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform  = test_transform

# DataLoader로 dataset을 mini-batch 단위로 묶어서 제공
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(dataset.classes)
print(dataset.class_to_idx)