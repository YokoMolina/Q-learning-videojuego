

import torch
import numpy as np
from Librerias.perceptron import SLP
# utilidades que se pueden programar y llamar 
# como funciones de decrecimiento
from utils.decay_schedule import LinearDecaySchedule
import random
import gym
from gym.spaces import Box
from utils.experience_memory import ExperienceMemory, Experience

MAX_NUM_EPISODE = 50
STEPS_PER_EPISODE = 30


class SwallowQLearner (object):
    def __init__(self, env, learning_rate = 0.005, gamma = 0.98):
        #inicialicemos los valores del objeto self
        self.learning_rate = learning_rate # podemos guardar el dato
        self.obs_shape = env.observation_space.shape
        
        self.action_shape = env.action_space.n

        low_act = -np.inf
        high_act = np.inf
        dtypes = np.float32

        self.action_space = Box(low=low_act, high=high_act, shape=(self.action_shape,), dtype=dtypes)

        # Define el tipo de dato para el espacio de observación
        low_obs = -np.inf
        high_obs = np.inf
        self.observation_space = Box(low=low_obs, high=high_obs, shape=self.obs_shape, dtype=dtypes)

        self.Q = SLP(self.obs_shape, self.action_shape)
        # se le pasa el espacio de observaciones 
        # el valor de salida es le espacio de acciones pues es lo que el agente desea saber para saber que hacer


        # se va a optimizar las ponderaciones de las entradas a travez de Adam
        self.Q_optimazer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate) # delvuelve los parametreso que es una funcion eredadda de la calse torch
        # lr es el ratio de aprendizaje que es un numero pequeño q es lo q aprende de una iteracion
        
        self.gamma = gamma

        self.epsilon_max = 1.0 #valor que se va incrementando si sobrepasa el epsiolon_min
        self.epsilon_mim = 0.05
        # funcion que ayuda a ir decayendo el epsilon 
        total_steps = MAX_NUM_EPISODE*STEPS_PER_EPISODE
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_mim,
                                                 max_steps = 0.5*total_steps)
        self.step_num = 0 # num de operacion
        # politica de actuacion
        self.polity = self.epsilon_greedy_Q

        # inicializamos la meroria
        self.memory = ExperienceMemory(capacity = int(1e5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    def get_action(self, obs):
        # retornar la politica dependiendo de la observacion
        return self.polity(obs)
    
    # la politica 
    # DEAIMIENTO DEL EPSILON IMPLEMENTADA EN UNA FUNCION 
    def epsilon_greedy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num): # funcion de decrecimiento que le paso el numero de iteracion
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device("cpu")).numpy())
        return action

    def learn(self, obs, action, reward, next_obs):
        # valor esperado de la calidad de la obs que se comrara con el output de la red 
        td_target = reward + self.gamma * torch.max(self.Q(next_obs)) # la salida será un tensor con las calidades y las acciones tomadas para cada calidad en la obs
        
        # diferencial temporal
        # resta la salida de la neurona con el valor objetivo que serra el diferencial temporal
        # funcion de costos
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        
        # 

        self.Q_optimazer.zero_grad() # se usa para limpiar los gradientes antes calculados 
        td_error.backward() # calcula los gradientes de la funcion de perdida con respecto a los parametrso del modelo
        # optimiza los pesos de la red con Adam
        self.Q_optimazer.step()

    # vamos a incluir la experiencia/ memoria
    def replay_experience(self, batch_size):
        #vuelve a jugar usando la experiencia aleatoria almacenada
        # param batch_size es el tamaño de la muestra a tomar de la memoria 
        # return nada

        experience_batch = self.memory.sample(batch_size)
        # nuevometodo de aprendizaje
        self.learn_from_batch_experience(experience_batch)
    

    
   # def learn_from_batch_experience(self, experiences):
        # extiende el memotod learn e incorpora las experiencias que pueden volver a usarse
        # Actualiza la red neuronal profunda en base a lo aprendido en el conjutno de 
        # experiencias anteriores
        #return nada
  #      batch_xp = Experience(*zip(*experiences)) # le apso como referencia con el *
  #      obs_batch = np.array(batch_xp.obs) # piuedo acceder a los parametros de la estructura
  #      action_batch = np.array(batch_xp.action)
  #      reward_batch = np.array(batch_xp.reward)
  #      next_obs_batch = np.array(batch_xp.next_obs)
  #      done_batch = np.array(batch_xp.done)

        # voy a ir remplazando de la funcion learn anterior por lo obtenido de la mamoria 
        # version vectorial 
        # diferencial temporal
        # si done es true sigo iterando
  #      td_target = reward_batch + ~ done_batch * np.tile(self.gamma, len(next_obs_batch)) * self.Q(next_obs_batch).detach().max(1)[0].data
        
  #      td_target = td_target.to(self.device)
  #      action_idx = torch.from_numpy(action_batch).to(self.device)
  #      td_error = torch.nn.functional.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1,1)), 
                                                # consulto todas las observaciones antorioes de la matriz Q en el estado dado
  #                                             td_target.float().unsqueeze(1) )
                                            # restamos la version pasada td_target
  #      self.Q_optimazer.zero_grad()
  #      td_error.mean().backward() # hago un paso para que aprenda
  #      self.Q_optimazer.step()


    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fframento de recuerdos anteriores
        :return:
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)
        
        td_target = reward_batch + ~done_batch * \
                    np.tile(self.gamma, len(next_obs_batch)) * \
                    self.Q(next_obs_batch).detach().max(1)[0].data.numpy()
        td_target = torch.from_numpy(td_target)
        td_target = td_target.to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        td_error = torch.nn.functional.mse_loss(
                    self.Q(obs_batch).gather(1,action_idx.view(-1,1).long()),
                    td_target.float().unsqueeze(1))
        
        self.Q_optimazer.zero_grad()
        td_error.mean().backward()
        self.Q_optimazer.step()


# metodo global de aprendizaje

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    #env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
   ## env = gym.make("BipedalWalker-v2")
    #env = gym.make("Acrobot-v1")
    agent = SwallowQLearner(env)
    first_episode = True
    episode_rewards = list()
    max_reward = float("-inf")
    for episode in range(MAX_NUM_EPISODE):
        obs = env.reset() # primera observacion
        total_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            env.render() # para verlo visual
            action = agent.get_action(obs)
            # extraigo los valores
            next_obs, reward, done, info = env.step(action)

            #almaceno en memoria todos los datos de la observacion 
            agent.memory.store(Experience(obs, action, reward, next_obs, done)) 
            
            batch_size = 32
            
           # agent.learn(obs, action, reward, next_obs)
            
            obs = next_obs
            total_reward += reward

# NO SON IID LAS OBSERVACIONES ES DECIR ESTAN RELACIONADAS LAS SIGUIENTES CON LA ANTERIRO
# SON ACCIONES SECUENCIALES DEPENDIENTES
# LA RED NEURONAL CONVERGE MÁS RAPIDO CON OBSERVACIONES IID
# ASI VAMOS A UTILIZAR EXPERIENCIAS PASADAS PARA PODER ESTIMAR LOS VALORS Q DE MEKOR MANERA Y MÁS RAPIDA 
# ES DECIR SE VA A CREAR UNA TUOLA EN DONDE SE VAN A IR GUARDANDO LAS EXPERIENCIAS 


#ctrls+ALTT+c
            # if done:
            #     env.reset()
            #     if first_episode:
            #         max_reward = total_reward
            #         first_episode = False
            #     episode_rewards.append(total_reward)
            #     if total_reward > max_reward:
            #         max_reward = total_reward
            #         print("\n Episodio {} finalizado con {} iteraciones. Recompenza = {}, Recompenza media = {}, mejor recompenza = {}".format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
            #         break

            if done:
                episode_rewards.append(total_reward)
                if total_reward > max_reward:
                    max_reward = total_reward
                print("\n Episodio {} finalizado con {} iteraciones. Recompensa = {}, Recompensa media = {}, mejor recompensa = {}".format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
                
                #AQUI VAMOS A JUGAR CON LA EXPERIENCIA (MEMORIA)
                if agent.memory.get_size() > 100:
                    agent.replay_experience(batch_size)
                    break
    env.close()
