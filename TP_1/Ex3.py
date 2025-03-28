import time

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()
print(f"Espace d'action :{env.action_space}")
print(f"Espace d'observation :{env.observation_space}")
step_count = 0
for _ in range(100):

    action = int(input("entre action : "))
    observation, reward, done, _, _ = env.step(action)
    print(f"Action :{action},Observation :{observation}, Reward :{reward}")
    print("done :",done)
    step_count +=1
    if done:
        env.reset()
        print(step_count)
        step_count = 0
env.close()
