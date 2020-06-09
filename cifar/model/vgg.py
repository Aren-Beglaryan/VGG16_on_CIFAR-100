import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import max, flatten


class MYVGG(nn.Module):
    
    def __init__(self,n_input_channels=3, n_output=100):
        super(MYVGG, self).__init__()
        
        self.conv1 = nn.Conv2d(n_input_channels,64,kernel_size=(3,3), padding=(1,1))
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,kernel_size=(3,3), padding=(1,1))
        self.conv2_bn = nn.BatchNorm2d(64)
        self.to_2_5 = nn.Conv2d(64,256,kernel_size=(1,1),stride=2)
        self.conv3 = nn.Conv2d(64,128,kernel_size=(3,3), padding=(1,1))
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,kernel_size=(3,3), padding=(1,1))
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256,kernel_size=(3,3),padding=(1,1))
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256,kernel_size=(3,3),padding=(1,1))
        self.conv6_bn = nn.BatchNorm2d(256)
        self.to_6_9 = nn.Conv2d(256,512,kernel_size=(1,1),stride=2)
        self.conv7 = nn.Conv2d(256,512,kernel_size=(3,3),padding=(1,1))
        self.conv7_bn = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512,512,kernel_size=(3,3),padding=(1,1))
        self.conv8_bn = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512,512,kernel_size=(3,3),padding=(1,1))
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512,512,kernel_size=(3,3),padding=(1,1))
        self.conv10_bn = nn.BatchNorm2d(512)
        self.to_10_12 = nn.Conv2d(512,512,kernel_size=(1,1),stride=1)
        self.conv11 = nn.Conv2d(512,512,kernel_size=(3,3),padding=(1,1))
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512,512,kernel_size=(3,3),padding=(1,1))
        self.conv12_bn = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512,512,kernel_size=(3,3),padding=(1,1))
        self.conv13_bn = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout2d(0.5)
        
        self.classifier = nn.Sequential(
            nn.Linear(512,4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(4096,n_output))
        
    def forward(self, x):
        # layer 1 (64,32,32)
        x = F.relu(self.conv1_bn(self.conv1(x)), inplace = True)
        
        # layer 2 (64,16,16)
        x = F.relu(self.conv2_bn(self.conv2(x)), inplace = True)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        y1 = self.to_2_5(x)
        
        # layer 3 (128,16,16)
        x = F.relu(self.conv3_bn(self.conv3(x)), inplace = True)
        
        # layer 4 (128,8,8)
        x = F.relu(self.conv4_bn(self.conv4(x)), inplace = True)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        # layer 5 (256,8,8)
        x = F.relu(self.conv5_bn(self.conv5(x)) + y1, inplace = True)#
        

        # layer 6 (256,8,8)
        x = F.relu(self.conv6_bn(self.conv6(x)), inplace = True)
        y2 = self.to_6_9(x)

        # layer 7 (512,4,4)
        x = F.relu(self.conv7_bn(self.conv7(x)), inplace = True)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # layer 8 (512,4,4)
        x = F.relu(self.conv8_bn(self.conv8(x)), inplace = True)

        # layer 9 (512,4,4)
        x = F.relu(self.conv9_bn(self.conv9(x)), inplace = True)
        x = F.relu(self.conv9_bn(self.conv9(x)) + y2, inplace = True)#
        
        # layer 10 (512,2,2)
        x = F.relu(self.conv10_bn(self.conv10(x)), inplace = True)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        y3 = self.to_10_12(x)

        # layer 11 (512,2,2)
        x = F.relu(self.conv11_bn(self.conv11(x)), inplace = True)

        # layer 12 (512,2,2)
        x = F.relu(self.conv12_bn(self.conv12(x)) + y3, inplace = True)#
        
        # layer 13 (512,1,1)
        x = F.relu(self.conv13_bn(self.conv13(x)), inplace = True)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = flatten(x,1)
        x = self.classifier(x)
        return x
    
    
    def predict(self, x):
        outputs = self.forward(x)
        _, predicted = max(F.softmax(outputs,dim=1).data, 1)
        return predicted