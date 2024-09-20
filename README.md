# ReinforcementLearning

- Implemented the simplest Vanilla Policy Gradient for a continuous action space based on OpenAI's Spinning Up documentation. 
  - Implemented it without pytorch/tensorflow because I wanted to know what was going on under the hood. Since I'd be calculating the policy gradient symbollically, I used and updated the simple_right.py environment. The simple_right.py environment just takes in a number as an action and spits out a reward. The reward is equal to the action's negative distance from the number 5 (e.g. distance between 6 and 5 is 1, so reward is -1. same with 4.).
  - The code is kinda messy. Lots of useless comments.
  - Calculations to get the policy gradient:
![image](https://github.com/user-attachments/assets/949ecce1-be88-446a-8531-15b70c4fed03)


Exercise for any beginners looking at this is to try a more complex policy. I initially had mu=b0 + b1s but it turns out that leads to NaN values. Try to explain why.


Keywords: Vanilla Policy Gradient, VPG, REINFORCE algorithm, simplest gradient policy, no baseline, symbollic differentiation, analytical differentiation, how to compute policy gradient,
