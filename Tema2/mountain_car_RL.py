# por ejemplo.. en montain car, se tiene un estacio de estados 2
    #donde la primera componenete es la posicion del carro y la otra es la veloicidad
    # en las acciones 0 es ir a la izquierda 1 es n o hacer nada y 2 es ir a la derecha

    # done será true cuando el carro este en posicin 0.5
    

# ELECCION RANDOM

import gym

# entorno

enviroment = gym.make("MountainCar-v0")

MAX_NUM_EPISODES = 1000

for episode in range(MAX_NUM_EPISODES):
    done = False
    # abrimos los ojos al agente
    obs = enviroment.reset()
    total_reward = 0.0 # para aguardar la recomopensa en cada episodio
    step = 0
    while not done: # mientras done no sea true
        enviroment.render() # creamos el entorno
        action = enviroment.action_space.sample() # accion aleatoria que se remplazara por la deciosiondel agemte inteligente
        # ejecutamos la accion
        next_state, reward, done, info = enviroment.step(action)
        total_reward += reward
        step += 1
        obs = next_state
    print("\n episodio número {} finalizado {} iteraciones. Recompeza final = {}".format(episode,step+1,total_reward))

enviroment.close()


