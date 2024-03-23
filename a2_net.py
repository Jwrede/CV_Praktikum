import torch
import torch.nn as nn


# Define the MLP class
class SimpleMLP(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(12, 128) 
        self.fc2 = nn.Linear(128, 256) 
        self.fc3 = nn.Linear(256, 1024) 
        self.fc4 = nn.Linear(1024, 256) 
        self.fc5 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
