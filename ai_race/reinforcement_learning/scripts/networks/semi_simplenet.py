'''
Reference:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.bn4 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))), stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))), stride=1)
        linear_input_size = convw * convh * 32
        #self.head = nn.Linear(linear_input_size, outputs)
        self.head = nn.Linear(linear_input_size + 1, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, angular_z):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        """
        print('x:', x.size())
        print('angular_z:', angular_z.size())
        print('x type:', x.dtype)
        print('angular_z type:', angular_z.dtype)
        """
        print('x:', x.size())
        print('angular_z:', angular_z.size())
        x = torch.cat([x, angular_z], dim=1)
        return self.head(x)
