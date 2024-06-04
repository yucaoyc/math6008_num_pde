import torch
import torch.nn as nn
import torch.nn.functional as F

# Code adapted from https://github.com/JiantingFeng/Deep-Ritz-PDE-Solver/blob/main/block.py

class PowerReLU(nn.Module):
    """
        Implements ReLU(x)^(power)
    """
    def __init__(self, power=3):
        super(PowerReLU, self).__init__()
        self.power = power

    def forward(self, input):
        return torch.pow(F.relu(input), self.power)
    
    
class DeepRitz(nn.Module):
    """
        Define the neural network class
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepRitz, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.act = PowerReLU()

    def forward(self, x):
        y = self.act(self.fc2(self.act(self.fc1(x)))) 
        y[:,0:self.input_size] += x
        y = self.act(self.fc4(self.act(self.fc3(y)))) + y
        y = self.fc5(y)
        return y