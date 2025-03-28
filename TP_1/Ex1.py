import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()
print(f"Espace d'action :{env.action_space}")
print(f"Espace d'observation :{env.observation_space}")
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(f"Action :{action},Observation :{observation}, Reward :{reward}")
    print("done :",done)
    if done:
        env.reset()
env.close()