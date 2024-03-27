import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        # Adjust the input size to match the flattened observation shape
        input_size = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),  # Adjusted first layer to accept flattened input
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Ensure observations are flattened
        observations = observations.view(observations.size(0), -1)
        return self.net(observations)
