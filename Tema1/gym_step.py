#NOTA: PODEMOS NOTAR QUE BIPEDAL NECESITA MAS ENTREMANIENTO PARA PODER CAMINAR
# LUNAR LLEGA MAS RAPIDO A LA META DEBIDO A LA CONSTRUCCION DEL ENTORNO
# ES UN JET QUE TIENE QUE ESTACIONARSE Y LA ESTACION ESTA EN EL CENTRO DEL
# ENTORNO  
#________________________________
# montain termina por tiempo
# bipedal por vidas
# lunar vidas xq tiene q caer en la linea de meta

#____________________________
import gym


#ENVIROMENTS
#_________________________________________________
#LA FUNCIÓN MAKE CREA EN AMBIENTE LLAMDO EN COMILLAS 
#enviroment = gym.make("MountainCar-v0")
#enviroment = gym.make("Qbert-v0")
MAX_NUM_EPISODE = 10
MAX_STEPS_PER_EPISODE = 500
#enviroment = gym.make("Acrobot-v1")
enviroment = gym.make("LunarLander-v2")

#____________________________________
#otros ambientes
#LIBRERIA BOX2D
#enviroment = gym.make("BipedalWalker-v3")
#LIBRERIA ATARI---- tiene confluctos
#enviroment = gym.make("SpaceInvaders-v0")



#SE GENERA UN BUCLE DE 2000 ITERACIONES 
for episode in range(MAX_NUM_EPISODE):
    # despertemos al agente
    obs = enviroment.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
    #CREAMOS EL RENDER DE EL AMBIENTE 
        enviroment.render() # persove el entorno el agente
        action = enviroment.action_space.sample()
        next_state, reward, done, info=enviroment.step(action)
        obs = next_state # tiene que volver a percibir el ambiente

        if done is True:
            print("\n Episodio #{} terminado en {} steps.".format(episode, step+1))
            break
enviroment.close()