# to test a model 
from stable_baselines3 import DQN
from RL.DQN.carenv import CarlaEnv

#update here
models_dir = "C:\CARLA\CARLA_0.9.15\WindowsNoEditor\ProjectFiles\RL\models\\1711205478"

env = CarlaEnv()
env.reset()

#and update here
model_path = f"{models_dir}\\8000.zip"
model = DQN.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        print(reward)