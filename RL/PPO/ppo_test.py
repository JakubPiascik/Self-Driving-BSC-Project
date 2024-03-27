# to test a model 
from stable_baselines3 import PPO
from ppo_carenv import CarlaEnv 
from stable_baselines3.common.vec_env import DummyVecEnv

#update here
models_dir = "C:\CARLA\CARLA_0.9.15\WindowsNoEditor\ProjectFiles\RL\PPO\ppo_models\\1711472022"

def make_env():
    def _init():
        return CarlaEnv()  # Initialize and return an instance of your environment
    return _init

# Wrap your environment with DummyVecEnv
env = DummyVecEnv([make_env()])  # This correctly vectorizes the environment

#and update here
model_path = f"{models_dir}\\160000.zip"
model = PPO.load(model_path, env=env)

episodes = 5
episode_summaries = []

for ep in range(episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #env.render()
        total_reward += reward  # Accumulate rewards
        step_count += 1  # Increment step count

    episode_summaries.append(f"Episode: {ep+1}, Total Reward: {total_reward}, Steps: {step_count}")

print("\n--- Episode Summaries ---")
for summary in episode_summaries:
    print(summary)