from stable_baselines3 import PPO
import os
import time
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_steeronly import CarlaEnv  # Ensure this is the correct import path for your environment


def main():
    print('This is the start of the training script')
    print('Setting folders for logs and models')

    models_dir = f"ppo_models/{int(time.time())}/"
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
    model = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001, tensorboard_log=logdir)

    TIMESTEPS = 5_000  # How long is each training iteration - individual steps
    iters = 0
    while iters < 4:
        iters += 1
        print(f'Iteration {iters} is to commence...')
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        print(f'Iteration {iters} has been trained')
        model.save(f"{models_dir}/{TIMESTEPS*iters}")

if __name__ == '__main__':
    main()
