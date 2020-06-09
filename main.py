import time
import copy


import torch
import torch.nn as nn
import torch.optim as optim


from cifar.model.vgg import MYVGG
from cifar.model.device import DEVICE

from cifar.loaders import train_loader, test_loader

from cifar.utils import get_accuracy

net = MYVGG()
net.to(DEVICE)


criterion = nn.CrossEntropyLoss()


epochs = 200
print_every = 49
running_loss = 0
best_accuracy_val = 0
best_model = None
nb_steps = len(train_loader)
accuracy_val = []
accuracy_tr = []
losses = []
learn_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr=learn_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70,90,110], gamma=0.1)
for e in range(epochs):
    start = time.time()


    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        
        if step % print_every == 0: 
            net.eval()
            with torch.no_grad():

                accuracy_tr.append(get_accuracy(net, train_loader, DEVICE))
                accuracy_val.append(get_accuracy(net, test_loader, DEVICE))
                
                if accuracy_val[-1] > best_accuracy_val: # Save best model
                    best_model = copy.deepcopy(net)
                    best_accuracy_val = accuracy_val[-1]

                print("Epoch: {}/{}".format(e+1, epochs),
                      "Step: {}/{}".format(step, nb_steps),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Train accuracy: {:.3f}%".format(accuracy_tr[-1]),
                      "Test accuracy: {:.3f}%".format(accuracy_val[-1]),
                      "{:.3f} s/{} steps".format((time.time() - start), print_every)
                     )
            losses.append(running_loss/print_every)
            running_loss = 0
            start = time.time()
            net.train()
            
print("Best reached accuracy was {}".format(best_accuracy_val))
