import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from collections import deque
import numpy as np

torch.manual_seed(102)
class PolicyNN(nn.Module):
# Try separating out mu and sigma
    def __init__(self, env, hidden_nodes):

        super().__init__()
        self.state_size = env.observation_space.shape[0]
        self.hidden_nodes = hidden_nodes
        # create parameterized policy
        self.mu_model = nn.Sequential(nn.Linear(self.state_size, self.hidden_nodes),
                                          nn.Tanh(),
                                          nn.Linear(self.hidden_nodes, 1))
        # from spinningup
        log_std = -0.5 * np.ones(env.action_space.shape[0], dtype = np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    def forward(self, state):

        state_torch = torch.from_numpy(state)
        state_torch.requires_grad = True
        mu = self.mu_model(state_torch)
        sigma = torch.exp(self.log_std)

        return mu, sigma


class VPGAgent():

    def __init__(self, env, discount_factor):
        """
        max_iterations: int
            Maximum number of times environment should run in an episode
        """
        self.env = env
        self.trajectories = []
        self.rewards = []
        self.discount_factor = discount_factor
        self.logprobs = []


    def get_action(self, state, policy):
        """
        Considers policy to be Gaussian distribution (from Sutton and Barto 2020)
        :param state:
        :return:
        """

        mu, sigma = policy(state)

        gaussian = Normal(mu, sigma)
        action_sample = gaussian.sample()

        # track log probabilities
        logprob = gaussian.log_prob(action_sample)
        self.logprobs.append(logprob)

        return np.array([action_sample.item()], dtype = 'f4'), mu.item(), sigma.item()

    def get_reward(self, t):
        """Calculate reward for an episode at certain time, t"""

        G = 0
        for i, r in enumerate(self.rewards[t:len(self.rewards)]):
              G += r * self.discount_factor**i

        return G

    def collect_memory(self, observation, reward):
        """Add each observation and reward to memory"""
        self.trajectories.append(observation)
        self.rewards.append(reward)

    def reset_memory(self):
        self.trajectories = []
        self.rewards = []
        self.logprobs = []