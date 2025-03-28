import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()
print(f"Espace d'action :{env.action_space}")
print(f"Espace d'observation :{env.observation_space}")
ValeurAction0=[]
ValeurAction1=[]
for _ in range(100):
    action = 0
    observation, reward, done, _, _ = env.step(action)
    print(f"Action :{action},Observation :{observation}, Reward :{reward}")
    ValeurAction0.append({"r":reward,"o":observation,"d":done})

    if done:
        env.reset()

for _ in range(100):
    action = 1
    observation, reward, done, _, _ = env.step(action)
    print(f"Action :{action},Observation :{observation}, Reward :{reward}")
    ValeurAction1.append({"r":reward,"o":observation,"d":done})

    if done:
        env.reset()
env.close()
for i in ValeurAction0:
    print(i.get("r"))

for i in ValeurAction1:
    print(i.get("r"))