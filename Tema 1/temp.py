# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym

enviroment = gym.make("MountainCar-v0")
enviroment.reset()
for _ in range(2000):
    enviroment.render()
    enviroment.step(enviroment.action_space.sample())
