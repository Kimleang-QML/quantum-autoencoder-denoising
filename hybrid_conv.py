import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

from torch_circuit import TorchCircuit

from constant import *
        
class ConvDenoiseNet(nn.Module):
    def __init__(self):
        super(ConvDenoiseNet, self).__init__()
        n_filters = 32
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, n_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc2 = nn.Linear(36, NUM_QUBITS * NUM_CIRCUITS*2)
        self.qc = TorchCircuit.apply
        self.out = nn.Linear(NUM_CIRCUITS, 36)

        ## decoder layers ##
        self.t_conv0 = nn.ConvTranspose2d(4, 4, 3, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(4, n_filters, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(n_filters, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        x = nn.Flatten()(x)
        
        x = self.fc2(x)
                
        x = torch.tanh(x)
        x = x.view(-1) 
        x = torch.cat([(x[i * 2 : i*2 + 2]) / torch.norm(x[i * 2: i*2 + 2]) for i in range(NUM_CIRCUITS*2)], dim=-1) # normalize sin and cos for each angle
        x = torch.stack([torch.atan2(x[i * 2], x[i*2 + 1]) for i in range(NUM_CIRCUITS*2)]) # convert to angles
        
        x = torch.cat([self.qc(x[i * 2 : i*2 + 2]) for i in range(NUM_CIRCUITS)], dim=1) # QUANTUM LAYER
        
        x = torch.Tensor(x.float())
        x = self.out(x)
        x = x.view(-1, 4, 3, 3)
        
        ## decode ##
        x = F.leaky_relu(self.t_conv0(x))        
        x = F.leaky_relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        return x
