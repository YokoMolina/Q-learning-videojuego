#shift+t
#IMPORTAMOS LA LIBRERIA GYM PARA EL APRENDIZAJE POR REFUERZO
import gym


#ENVIROMENTS
#_________________________________________________
#LA FUNCIÓN MAKE CREA EN AMBIENTE LLAMDO EN COMILLAS 
#enviroment = gym.make("MountainCar-v0")
##enviroment = gym.make("CartPole-v1")
#enviroment = gym.make("Acrobot-v1")
#enviroment = gym.make("Pendulum-v0")

#____________________________________
#otros ambientes
#enviroment = gym.make("BipedalWalker-v3")

enviroment = gym.make("SpaceInvaders-v0")



#RESETEAMOS EL AMBIENTE
enviroment.reset()

#SE GENERA UN BUCLE DE 2000 ITERACIONES 
for _ in range(2000):
    #CREAMOS EL RENDER DE EL AMBIENTE 
    enviroment.render()
    #HACEMOS SE TOME DECISIONES ALEATORIAS DE TODAS LAS POSIBLES 
    enviroment.step(enviroment.action_space.sample())

enviroment.close
    
