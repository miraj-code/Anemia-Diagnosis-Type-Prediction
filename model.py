# Import the necessary liraries to build the model
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a class for the model
class Model(nn.Module):
    # Initialize the layers of the model. The model has 1 input layer (with 14 neurons), 
    # 3 hidden layers (each with 15 neurons), and 1 output layer (with 9 neurons)
    def __init__(self, in_features = 14, h1 = 25, h2 = 25, h3 = 25, out_features = 9):
        super().__init__()
        # Create 5 fully connected layers
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    # Forward function to pass/forward data through the fully connected layers
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x

# Create a manual seed for randomization
torch.random.seed()

# Initialize the model
model = Model()
