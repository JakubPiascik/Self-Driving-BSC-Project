import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        # Initialize a sequential container of layers: two hidden layers and an output layer with ReLU activations
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),  # First layer with input size and 64 neurons
            nn.ReLU(),  # Activation function to introduce non-linearity
            nn.Linear(64, 64),  # Second hidden layer with 64 neurons
            nn.ReLU(),  # Activation function
            nn.Linear(64, features_dim),  # Output layer with features_dim neurons
            nn.ReLU()  # Activation function
        )

    # Define the forward pass through the network
    def forward(self, observations):
        return self.net(observations)  # Return the network's output
