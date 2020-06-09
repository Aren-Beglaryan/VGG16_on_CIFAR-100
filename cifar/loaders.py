import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from cifar.resources import (
    train_data, train_label,
    test_data, test_label
)
from cifar.dataset import CustomDataset


train_data, test_data = train_data / 255, test_data /255

transform_train = transforms.Compose(
        [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]
    )

train_dataset = CustomDataset(train_data, train_label, transform_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    pin_memory=torch.cuda.is_available(),
    shuffle=True
)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
test_dataset = CustomDataset(test_data, test_label, transform_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    pin_memory=torch.cuda.is_available()
)