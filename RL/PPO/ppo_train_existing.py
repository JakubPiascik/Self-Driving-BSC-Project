# This is identical to train.py but loads in a previously trained model using models_dir

from stable_baselines3 import PPO
import os
import time
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_steeronly import CarlaEnv  # Ensure this is the correct import path for your environment


def main():
    print('This is the start of the training script')
    print('Setting folders for logs and models')

    models_dir = "C:\CARLA\CARLA_0.9.15\WindowsNoEditor\ProjectFiles\RL\PPO\ppo_models\\1711556560"
    logdir = f"logs/{int(time.time())}/"

    try:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
    except Exception as e:
        print(f"Error creating directories: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Connecting to environments...')


    env = CarlaEnv()
    print('Env action space:',env.action_space)
    env.reset()
    print('Env has been reset as part of launch')

    initial_model_name = "520000"
    model_path = f"{models_dir}\\520000"
    model = PPO.load(model_path, env=env)
    ('Model has been loaded')

    TIMESTEPS = 40_000  # How long is each training iteration - individual steps
    current_timesteps = int(initial_model_name)

    iters = 0
    while iters < 25:
        iters += 1
        print(f'Iteration {iters} is to commence...')
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        print(f'Iteration {iters} has been trained')
        current_timesteps += TIMESTEPS
        model.save(os.path.join(models_dir, str(current_timesteps)))

if __name__ == '__main__':
    main()
