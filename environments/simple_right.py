# A very simple environment where you get higher reward when you choose a higher positive number
# The state is updated based on the action and is capped at 10. 

import gym
from gym import spaces
import numpy as np

class SimpleRight(gym.Env):

    def __init__(self):
        super(SimpleRight, self).__init__()

        self.observation_space = spaces.Box(low = -10,
                                            high = 10,
                                            shape = (1, ))


        self.action_space = spaces.Box(low = 0,
                                       high = 10,
                                       shape = (1,))

        # start in the middle
        self.state = np.array([0], dtype = 'f4')

        self.total_reward = 0
        self.counter = 0

    def step(self, action):

        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        new_state = np.clip(self.state + action, self.observation_space.low, self.observation_space.high)
        self.state = new_state

        if self.counter == 50:
            done = True
        else:
            self.counter += 1
            done = False
        reward = action[0]
        self.total_reward += reward
        #dummy for now
        info = None

        return self.state, reward, done, info



    def reset(self, seed = 42, return_info = True):

        info = None
        self.state = np.array([0], dtype = 'f4')
        self.total_reward = 0
        self.counter = 0
        return self.state, info


    def render(self):

        print(f'Current position is: {self.state[0]}')
        print(f'Accrued reward is: {self.total_reward}')
        print('===========================================')


if __name__ == "__main__":
    # Test run of the environment
    env = SimpleRight()
    obs, info = env.reset()
    env.render()
    for i in range(10):
        action = env.action_space.sample()
        print(f'Action:{action}')
        obs, reward, done, info = env.step(action)
        env.render()
