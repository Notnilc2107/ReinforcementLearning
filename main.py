import gym
from vpg import VPGAgent
import matplotlib.pyplot as plt
import wandb

#wandb.init(project='VPG-test')

env = gym.make('Pendulum-v1', g=9.81)

observation, info = env.reset(seed=42, return_info=True)
max_iterations = 200
episodes = 500
hidden_nodes = 128
discount_factor = 0.9
learning_rate = 0.01

# wandb.config = {
#     'learning_rate': learning_rate,
#     'epochs': episodes,
#     'iterations': max_iterations,
#     'discount_factor': discount_factor
# }

agent = VPGAgent(env, max_iterations, hidden_nodes, discount_factor, learning_rate)

total_rewards = []

#training loop

for ep in range(episodes):
    for i in range(max_iterations):
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        agent.collect_memory(observation, reward)
        if done or i == (max_iterations - 1):
            observation, info = env.reset(return_info= True)

    total_rewards.append(sum(agent.rewards))
    #wandb.log({'rewards': sum(agent.rewards)})
    if ep % 25 == 0:
        print('Episode:', ep)
        print('Collected reward:', sum(agent.rewards))

    # calculate G for each time step to get policy gradient
    G = [agent.get_reward(i) for i in range(len(agent.rewards))]
    loss = [- agent.logprobs[i] * G[i] for i in range(len(agent.rewards))]
    total_loss = sum(loss)
    total_loss.backward()
    agent.optimizer.step()
    agent.optimizer.zero_grad()

    agent.reset_memory()
    env.close()

plt.plot(total_rewards)
plt.show()

## Next steps
## Implement get_reward() - figure out equation
## Then first implement without the advantage
## And without the value function regression (I think this is needed for implementing advantage)

## And then if I feel like it, implement with advantage (baseline)

## But maybe I can just plug this into the Citylearn environment and see how well it works.
## Theoretically, it should be plug-in-able!