# Test the set up of OpenAI Gym env
# HOGN SAN WONG



import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action





'''
Notes from OpenAI website
env.step() => is a function that returns 4 values
(1) Observation (Object) Represent the observation of the env. (Ex: pixel data from a camera, Joint angles)
(2) Reward (Float) Amount of reward achieved by PREVIOUS action (Goal: Increase REWARD)
(3) Done (Boolean) To determin whether it's time to reset the env. When DONE == True => episode has terminated
(4) info (dict) Diagnostic info useful for debug

  |-------------action ------->
AGENT			             ENV
  <---Observation, Reward-----|

AGENT - ENV - loop
Each timestep: Agent chooses an action, env return an observation and reward
Process gets start


'''