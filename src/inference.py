import gym
from stable_baselines3 import PPO, A2C
from config import params


env = gym.make('LunarLander-v2')
env.reset()

model_path = f'{params["inference_model_path"]}'
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, info = env.step(action)

env.close()