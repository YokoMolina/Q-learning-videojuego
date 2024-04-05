
# espacios / valores que puede tomar una accion
# box / R^{n} / 

#gym.spaces.Box(low=-10, high=10, shape=(2,))

# discrete / números enteros entre 0 y n-1
#gym.spaces.Discrete(5)

# dict/diccionarios / diccionario para espacios mas complejos
#gym.space.Dict({
 #   "position":gym.space.Discrete(3) #(0,1,2)
  #  "velocity":gym.space.Discrete(2) #(0,1)
#})

# multi binario (T,F)^{n}
#gym.space.MultiBinary(3)

# multi discreto / (a,a+1,a+2...)^{m}
# gym.space.MultiDiscrete([-10,10],[0,1])

# tuple / producto de espacios simples
#gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(2))) #(0,1,2)x(0,1)

#pring / randow speed
import gym
from gym.spaces import *
import sys

def print_spaces(space):
    print(space)
    if isinstance(space,Box): # vemos si el espacio suministrado es tipo Box
        print("\n Cota inferior: ", space.low)
        print("\n Cota superior: ", space.high)


if __name__ == "__main__":
    enviroment = gym.make(sys.argv[1])
    print("Espacio de estados: ")
    print_spaces(enviroment.observation_space)
    print("espacio de acciones: ")
    print_spaces(enviroment.action_space)
    try:
        print("descripcion de las acciones: ", enviroment.unwrapped.get_action_meanings(1))
    except AttributeError:
        pass

# por ejemplo.. en montain car, se tiene un estacio de estados 2
    #donde la primera componenete es la posicion del carro y la otra es la veloicidad
    # en las acciones 0 es ir a la izquierda 1 es n o hacer nada y 2 es ir a la derecha

    # done será true cuando el carro este en posicin 0.5
    