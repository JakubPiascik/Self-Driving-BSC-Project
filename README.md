# Autonomous Vehicle Navigation with DQN and PPO

## Description
This repository contains the implementation of two reinforcement learning algorithms, Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), applied to autonomous vehicle navigation in the CARLA simulation environment. The project's objective is to evaluate the learning efficiency of these models in terms of average reward and loss values.

## Features
- Integration with CARLA, an open urban driving simulator.
- Custom reward system to evaluate autonomous navigation.
- Iterative training loops with policy updates.
- Training and Testing scripts to evaluate DQN and PPO performance
- Implemntation of CNN into observation space

## Getting Started

### Dependencies
- Python 3.7 or higher
- [CARLA Simulator](https://github.com/carla-simulator/carla/releases)
- [TensorFlow](https://www.tensorflow.org/install)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

### Installing
1. Follow steps from CARLA repo to install on machine https://github.com/carla-simulator/carla
2. Clone the repository : git clone https://github.com/JakubPiascik/ProjectFiles.git
3. Install required Python packages: pip install -r requirements.txt

### Executing Program
1. Navigate to CarlaUE4.exe and execute it.
2. Run ppo_train.py to begin training.
3. Tensorboard logs will be stored under a directory called 'logs'
4. Run tensorboard --logdir=your_log_dir to observe visualisations and metrics from tensorboard
5. If you wish to test a trained model, modification of the directory is required in ppo_test.py

   
## Authors
- Jakub Piascik
- Piascik80@gmail.com

## Citations
@inproceedings{Dosovitskiy17,
  title = {{CARLA}: {An} Open Urban Driving Simulator},
  author = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages = {1--16},
  year = {2017}
}
