import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()
print(f"Espace d'action :{env.action_space}")
print(f"Espace d'observation :{env.observation_space}")
step_count = 0
nbrEpisod=0

action=0

for _ in range(100):
    action=env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(f"Action :{action},Observation :{observation}, Reward :{reward}")
    step_count =step_count+1
    if done:
        nbrEpisod =nbrEpisod+1
        env.reset()
env.close()
print ("La moyenne :", step_count/nbrEpisod)
print("nombre Episod",nbrEpisod)