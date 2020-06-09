import pickle
import numpy as np


def unpickle(file_path: str):
    with open(file_path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d

def reshape_data(data_dict: dict):
    l = []
    for elem in data_dict[b'data']:
        # https://github.com/weiaicunzai/pytorch-cifar100/blob/master/dataset.py
        r = elem[:1024].reshape(32, 32)
        g = elem[1024:2048].reshape(32, 32)
        b = elem[2048:].reshape(32, 32)
        image = np.dstack((r, g, b))
        l.append(image) 
    return np.array(l)

def get_accuracy(model, dataloader, device):
    total = 0
    correct = 0
    for (images, labels) in iter(dataloader):
        images, labels = images.to(device), labels.to(device)
        predicted = model.predict(images)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total