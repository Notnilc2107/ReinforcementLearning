# ReinforcementLearning

- Implemented the Vanilla Policy Gradient with no baseline for a continuous action space based on OpenAI's Spinning Up documentation. 
  - Implemented without pytorch/tensorflow because I wanted to know what was going on under the hood. Since I'd be calculating the policy gradient symbollically, I used a simple environment that takes in a number as an action and spits out a reward equal to the negative of the action's distance from the number 5 (e.g. distance between 6 and 5 is 1, so reward is -1. same with 4.). The environment belongs to https://github.com/mianakajima/ReinforcementLearning. All I did was change the reward and update it so that it works with gymnasium. It is essentially just a continuous version of the multi-armed bandit problem.
  - The code is kinda messy. Lots of useless comments.
  - Calculations to get the policy gradient:
![image](https://github.com/user-attachments/assets/949ecce1-be88-446a-8531-15b70c4fed03)
  - The highest reward is when action=5, so Beta_0 should approach 5 (which it does).

Exercise for any beginners looking at this is to try a more complex policy. I initially had mu=b0 + b1s but it turns out that leads to NaN values. Try to explain why. Let me know if you do cus idk why.


Keywords: Vanilla Policy Gradient, VPG, REINFORCE algorithm, simplest gradient policy, no baseline, symbollic differentiation, analytical differentiation, how to compute policy gradient, reinforcement learning no neural network, reinforcement learning without neural network,
