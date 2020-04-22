## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from collections import OrderedDict


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=5, padding=2)),
            ('relu1', nn.ReLU())
            ]))
        
        self.maxp1 = nn.Sequential(OrderedDict([
                ('maxp1', nn.MaxPool2d(2, 2)),
                ('dropout1', nn.Dropout(0)),
                ('bachnorm1', nn.BatchNorm2d(32))
                ]))
        
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(32, 64, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU())
            ]))
        
        self.maxp2 = nn.Sequential(OrderedDict([
                ('maxp2', nn.MaxPool2d(2, 2)),
                ('dropout2', nn.Dropout(0)),
                ('bachnorm2', nn.BatchNorm2d(64))
                ])) 
        
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(64, 128, kernel_size=5, padding=2)),
            ('relu3', nn.ReLU())
            ]))
        
        self.maxp3 = nn.Sequential(OrderedDict([
                ('maxp3', nn.MaxPool2d(2, 2)),
                ('dropout3', nn.Dropout(0)),
                ('bachnorm3', nn.BatchNorm2d(128))
                ]))
        
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 * 28 * 28, 1024)),
            ('relu5', nn.ReLU()),
            ('dropout5', nn.Dropout(0)),
            ('bachnorm5', nn.BatchNorm1d(1024))
            ])) 
        
        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(1024, 1024)),
            ('relu6', nn.ReLU()),
            ('dropout6', nn.Dropout(0)),
            ('bachnorm6', nn.BatchNorm1d(1024))
            ]))
        
        self.fc3 = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(1024, 136))
            ]))
        
       
    
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
