# NO SON IID LAS OBSERVACIONES ES DECIR ESTAN RELACIONADAS LAS SIGUIENTES CON LA ANTERIRO
# SON ACCIONES SECUENCIALES DEPENDIENTES
# LA RED NEURONAL CONVERGE MÁS RAPIDO CON OBSERVACIONES IID
# ASI VAMOS A UTILIZAR EXPERIENCIAS PASADAS PARA PODER ESTIMAR LOS VALORS Q DE MEKOR MANERA Y MÁS RAPIDA 
# ES DECIR SE VA A CREAR UNA TUOLA EN DONDE SE VAN A IR GUARDANDO LAS EXPERIENCIAS 
# es decir vamos a crear la memoria del agente

from collections import namedtuple
import random


Experience = namedtuple("Experience", ["obs","action","reward","next_obs", "done"])
# generamos la estructura que tendrá todos estos parametros [ ]
# ESTO ES UN BUFFER 
class ExperienceMemory(object):
    # vamos a reproducir las experiencias , es decir, recuperarlas 
    # ESTO SERÁ UN BUFFER QUE SIMULA LA MEMORIA DEL AGENTE
    def __init__(self, capacity = int(1e6)):
        # :parame capacity es la capacidad total de mempria ciclica elimina memoria inicial que no sirve
        # numero max de experiencias almacenables 
        self.capacity = capacity
        self.memory_idx = 0 # identificador que sabe la experiencia actual #es el ultimo indice vacio

        self.memory = []

    def sample(self, batch_size):
        #batch:size es el tamaño de la memoria a recuperar
        # devuelve una muestra aleatoria del tamñano bacht_size de experiencias aleatorias  de la memoria
        assert batch_size <= self.get_size(), "el tamaño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory, batch_size) # extraigo de forma uniforme muestras del experience de forma aleatoria de tamaño batch_size

    def get_size(self):
        #return: numero de experiencias almacenadas en memoria 
        return len(self.memory) # tamaño de la memoria 
    
    def store(self, exp):
        #exp: objeto experiencia a ser almanecado en memoria 
        self.memory.insert(self.memory_idx % self.capacity, exp)
        # insertamos en la experiencia exp en memory_idx y el modulo % nos va a ayudar para que sea ciclica 
        # el guardar informacion es decir este modulo dará cero en varias veces lo cual actualizará el valor inicial
        self.memory_idx += 1