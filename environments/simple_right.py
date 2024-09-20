# If you've followed along with the gymnasium make your own environment where they make gridworld just put this in the gym_examples folder and register
# the environment by adding:
# register(
#      id="gym_examples/SimpleRight-v0",
#      entry_point="gym_examples.envs.simple_right:SimpleRight",
#      max_episode_steps=300,
# )
# to __init__.py in the gym_examples folder.

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleRight(gym.Env):

    def __init__(self):
        super(SimpleRight, self).__init__()

        self.observation_space = spaces.Box(low = -100,
                                            high = 100,
                                            shape = (1, ))


        self.action_space = spaces.Box(low = -100,
                                       high = 100,
                                       shape = (1,))

        # start in the middle
        self.state = np.array([0], dtype = 'f4')

        
        self.counter = 0

    def step(self, action):

        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        new_state = np.clip(self.state + action, self.observation_space.low, self.observation_space.high)
        self.state = new_state

        if self.counter == 50:
            truncated = True
        else:
            self.counter += 1
            truncated = False
        reward = - abs(action[0] - 5) # Best reward is when action[0]=5
        
        #dummy for now
        info = {}

        return self.state, reward, False, truncated, info



    def reset(self, seed = None, options = None):

        info = {}
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
