import gym
from stable_baselines3 import PPO, A2C
import os
import mlflow
from stable_baselines3.common.logger import HumanOutputFormat, Logger
import sys
from utils import MLflowOutputFormat
from config import params

ALGO_TYPE='PPO'
EXPT_NAME = 'Different Algorithms'

models_dir = f'{params["model_path"]}/{ALGO_TYPE}'
logs_dir = 'logs'

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)


if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

if __name__ == '__main__':

    mlflow.set_experiment(EXPT_NAME)

    mlflow.start_run(run_name=f"{ALGO_TYPE}-01")

    env = gym.make('LunarLander-v2')
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir, device='cpu')
    
    model.set_logger(loggers)
    env.reset()


    TIMESTEPS = 10000
    for i in range(params["epochs"]):
        model.learn(
            total_timesteps=TIMESTEPS, 
            reset_num_timesteps=False, 
            tb_log_name=f'{ALGO_TYPE}'
            )
        model.save(f'{models_dir}/{TIMESTEPS*i}')
        mlflow.log_artifact(f'{models_dir}/{TIMESTEPS*i}.zip', artifact_path="models")
        

    mlflow.log_param("algo_type", ALGO_TYPE)

    mlflow.end_run()
    env.close()