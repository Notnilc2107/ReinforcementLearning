# ReinforcementLearning

- Tried to implement the simplest Vanilla Policy Gradient for a continuous action and state space based on Ch.13 of Sutton and Barto and OpenAI's Spinning Up documentation. 
  - Tested this on `Pendulum-v1` OpenAI Gym environment but the agent couldn't learn to increase reward. 
  - I thought, maybe this simple VPG isn't powerful enough to solve the Pendulum environment. So I created a very simple environment where you choose a number between 0 and 10 (continuous) and you get the reward equivalent to the value chosen but got this:
  
  ![image](https://user-images.githubusercontent.com/96712795/186735196-701d0618-0a2d-4629-8fb2-55bdff66f081.png)

That looks okay, except that I would have expected it to plateau at 10 instead of around 5. This requires further debugging, but I'm going to put this on hold for now and pause work. 

Next directions I would go in if I am to continue this: 

- Possibly do an optimal learning rate search to check if the plateau is caused by faulty algorithm rather than just a bad learning rate
- Create a simpler environment where there is no observation 
  - This [blog post](https://andyljones.com/posts/rl-debugging.html) on debugging RL was very informational and also suggested the above. 
- The simple environment shouldn't need a neural network, so also try taking that out 
- Create more unit tests 
- Run Spinning Up's VPG implementation on the `Pendulum-v1` environment 

 
