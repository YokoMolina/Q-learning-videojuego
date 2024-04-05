import gym
import sys

def run_gym_enviroments(argv):
    # arg es un vector que tendrá como primer elemento será el nombre
    # del juego y luego el # de iteraciones a ejecutarse
    enviroment= gym.make(argv[1])
    enviroment.reset()
    for _ in range(int(argv[2])):
        enviroment.render()
        enviroment.step(enviroment.action_space.sample())
    
    enviroment.close()

if __name__ == "__main__":
    run_gym_enviroments(sys.argv)