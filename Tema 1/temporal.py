import gym



enviroment = gym.make("MountainCar-v0")
#enviroment = gym.make("CartPole-v1")
#enviroment = gym.make("Acrobot-v1")
#enviroment = gym.make("Pendulum-v0")
enviroment.reset()

for _ in range(2000):
    enviroment.render()
    enviroment.step(enviroment.action_space.sample())
