import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from collections import deque
import numpy as np

class VPGAgent():

    def __init__(self, env, max_iterations, hidden_nodes, discount_factor, lr):
        """
        max_iterations: int
            Maximum number of times environment should run in an episode
        """
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.trajectories = []
        self.rewards = []
        self.hidden_nodes = hidden_nodes
        self.discount_factor = discount_factor
        self.lr = lr
        self.logprobs = []

        # create parameterized policy
        self.policy_model = nn.Sequential(nn.Linear(self.state_size, self.hidden_nodes),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_nodes, 2))

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), self.lr)

    def get_action(self, state):
        """
        Considers policy to be Gaussian distribution (from Sutton and Barto 2020)
        :param state:
        :return:
        """
        state_torch = torch.from_numpy(state)
        actions = self.policy_model(state_torch)

        mu, sigma = actions[0], actions[1].exp()

        gaussian = Normal(mu, sigma)
        action_sample = gaussian.sample()

        # clip in case sample is not in action space
        clipped_action = torch.clip(action_sample, self.env.action_space.low[0], self.env.action_space.high[0]).item()

        # track log probabilities
        logprob = gaussian.log_prob(action_sample)
        self.logprobs.append(logprob)

        return np.array([clipped_action])

    def get_reward(self, t):
        """Calculate reward for an episode at certain time, t"""

        G = 0
        for i, r in enumerate(self.rewards[(t + 1):len(self.rewards)]):
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