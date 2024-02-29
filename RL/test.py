from stable_baselines3 import DQN
import os
from carenv import CarlaEnv
import torch
from custom_policy import CustomMLP

def load_model(model_path):
    """
    Function to load the trained model from a given path.
    """
    model = DQN.load(model_path)
    return model

def evaluate_model(model, env, num_episodes=10):
    """
    Function to evaluate the model over a specified number of episodes.
    Returns the average reward.
    """
    total_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
    average_reward = total_reward / num_episodes
    return average_reward

def main():
    print('Testing script initiated')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # Specify the path to your model directory
    models_dir = "path/to/your/saved/models/directory"

    # Load the environment
    env = CarlaEnv()
    
    # Assuming you have saved your model with a specific iteration count
    # Replace this with how you have named your saved models
    model_path = os.path.join(models_dir, "your_model_name")
    
    # Load the trained model
    model = load_model(model_path)
    
    # Evaluate the model
    num_episodes = 5  # You can modify this to run more or fewer episodes
    average_reward = evaluate_model(model, env, num_episodes=num_episodes)
    print(f'Average Reward over {num_episodes} episodes:', average_reward)

if __name__ == "__main__":
    main()