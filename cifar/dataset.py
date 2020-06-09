import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).transpose(3, 1).transpose(2, 3).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = transforms.ToPILImage(mode='RGB')(x)
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)