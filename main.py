import gym
from vpg import VPGAgent
from vpg import PolicyNN
import torch
import matplotlib.pyplot as plt
from environments.simple_right import SimpleRight
import wandb
import numpy as np

#wandb.init(project='VPG-test')

#env = gym.make('Pendulum-v1')

env = SimpleRight()

max_iterations = 50
episodes = 800
hidden_nodes = 16
discount_factor = 0.9
learning_rate = 0.01

# wandb.config = {
#     'learning_rate': learning_rate,
#     'epochs': episodes,
#     'iterations': max_iterations,
#     'discount_factor': discount_factor
# }

agent = VPGAgent(env, discount_factor)
policy_nn = PolicyNN(env, hidden_nodes)
optimizer = torch.optim.Adam(policy_nn.parameters(), learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
total_rewards = []
mu_all = []
sigma_all = []
action_all = []
losses = []
G_all = []
logprob_all = []

#training loop
torch.autograd.set_detect_anomaly(True)

for ep in range(episodes):
    observation, info = env.reset(return_info=True)
    mu_ep = []
    sigma_ep = []
    for i in range(max_iterations):
        action, mu, sigma = agent.get_action(observation, policy_nn)
        action_all.append(action[0])
        mu_ep.append(mu)
        sigma_ep.append(sigma)
        observation, reward, done, info = env.step(action)
        agent.collect_memory(observation, reward)
        if done:
            break

    total_rewards.append(sum(agent.rewards)/len(agent.rewards))
    mu_all.append(sum(mu_ep)/len(mu_ep))
    sigma_all.append(sum(sigma_ep)/len(sigma_ep))
    #wandb.log({'rewards': sum(agent.rewards)})
    if ep % 25 == 0:
        print('Episode:', ep)
        print('Collected avg reward:', total_rewards[-1])

    # calculate G for each time step to get policy gradient
    G = np.array([agent.get_reward(i) for i in range(len(agent.rewards))])
    #G = (G - G.mean())/G.std()

    loss = [- agent.logprobs[i] * G[i] for i in range(len(agent.rewards))]
    G_all.append(G.mean())
    total_loss = sum(loss)/len(loss)
    losses.append(total_loss.item())
    total_loss.backward()
    #torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)
    # for p in policy_nn.parameters():
    #     print(f'Ep {ep}: {p.grad}')
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    agent.reset_memory()

env.close()

plt.plot(total_rewards)
plt.title('Cumulative Reward')
plt.show()

plt.plot(mu_all)
plt.title('Mu')
plt.show()

plt.plot(sigma_all)
plt.title('Sigma')
plt.show()

plt.plot(action_all)
plt.title('Action')
plt.show()

plt.plot(losses)
plt.title('Total Losses')
plt.show()

plt.plot(G_all)
plt.title('Average G')
plt.show()

plt.plot(logprob_all)
plt.title('Average logprob')
plt.show()

## Next steps
## Implement get_reward() - figure out equation
## Then first implement without the advantage
## And without the value function regression (I think this is needed for implementing advantage)

## And then if I feel like it, implement with advantage (baseline)

## But maybe I can just plug this into the Citylearn environment and see how well it works.
## Theoretically, it should be plug-in-able!