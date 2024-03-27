from stable_baselines3.common.env_checker import check_env

from ppo_carenv import CarlaEnv

env = CarlaEnv()  # Replace with your environment
check_env(env)
