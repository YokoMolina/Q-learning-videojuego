

import torch
import numpy as np
import random
import gym
from gym.spaces import Box
import datetime
#_____________________-
from Librerias.perceptron import SLP
# utilidades que se pueden programar y llamar 
# como funciones de decrecimiento
from utils.decay_schedule import LinearDecaySchedule
from utils.experience_memory import ExperienceMemory, Experience
#__________________________________-
from  Librerias.cnn import CNN
from utils.params_manager import ParamsManager

# Parametrso globales

manager = ParamsManager("parameters.json")
#ficheros de la configuracion de las ejecuciones
summary_filename_prefix = manager.get_agent_params()["summary_filename_prefix"]
summary_filename = summary_filename_prefix + "ENTORNO_IRA_AQUI" + datetime.now().strftime("%y-%m-%d-%H-%M")
manager.export_agent_params(summary_filename + "/" + "agent_params.json")
manager.export_environments_params(summary_filename + "/" + "environment_params.json")
# contador global de ejecuciones
global_step_num = 0
# habilidatar entrenamiento po gpu o cpu
use_cuda = manager.get_agent_params()["use_cuda"]
device = torch.device("cuda:"+ str("id_de_gpu")) if torch.cuda.is_availablre() and use_cuda else "cpu" 
# habilitar la semilla aleatoria para poder reposucir  el experiemnto despues
seed = manager.get_agent_params()["seed"]                                      
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_avalable() and use_cuda:
    torch.cuda.manual_seed_all(seed)
                                          





class SwallowQLearner (object): # ya nos le vamos a pasar el environment
    def __init__(self, obs_shape, action_shape, params):
        
        self.params = params
        self.gamma = self.params["gamma"]
        self.learning_rate = self.params["learning_rate"]
        self.best_reward_mean = -float("inf")
        self.best_reward = -float("inf")
        self.training_steps_completed = 0
        

        # vamos a decidir si el objeto de entrada es una imagen o no y utilizar el perceptron o la cnn ya creada
        if len(obs_shape) == 1: # solo tenemos una dimension del espacio de observaciones
            self.DQN = SLP # metodo utilizado
        elif len(obs_shape) == 3: # el estado de observaciones es una imagen 3d
            self.DQN = CNN
        
        self.Q = self.DQN(obs_shape, action_shape, device).to(device)
        self.Q_optimazer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate) # delvuelve los parametreso que es una funcion eredadda de la calse torch
        
        if self.params["use_target_network"]: # vamos a actualizar a los ultimos valores 
            self.Q_target = self.DQN(obs_shape, action_shape, device).to(device)
        
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = self.params["epsilon_max"]
        self.epsilon_min = self.params["epsilon_min"]

        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min,
                                                 max_steps = self.params["epsilon_decay_sinal_step"])
        self.step_num = 0 # num de operacion
        # politica de actuacion
        self.polity = self.epsilon_greedy_Q

        # inicializamos la meroria
        self.memory = ExperienceMemory(capacity = int(self.params["experience_memory_size"]))

        

    def get_action(self, obs): # vamos a adecuar q vengan imagenes 
        # la observacion ya no va a ser un vector sino una imagen
        obs = np.array(obs)
        # 0 a 255
        obs = obs/255.0
        if len(obs.shape) == 3: # tenemos una imagen
            if(obs.shape[2] < obs.shape[0]): #anchuraxalturaxcolor
                # yo quiero que sea el colorx alturaxanchura
                obs = obs.reshape(obs.shape[2], obs.shape[1], obs.shape[0])
            obs = np.expand_dims(obs, 0) # vamos a expandir la imagen
        return self.polity(obs)
    
    # la politica 
    # DEAIMIENTO DEL EPSILON IMPLEMENTADA EN UNA FUNCION 
    def epsilon_greedy_Q(self, obs):
        self.step_num += 1 # el parametro test sera por consola y lo que hara el agentr es saltarse el modo cuando aleatorio del aprendizaje (acciones aleatorias)
        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]: # funcion de decrecimiento que le paso el numero de iteracion
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device("cpu")).numpy())
        return action

    def learn(self, obs, action, reward, next_obs, done): # funcion original de aprendizaje 
        if done: # si emos acabado el episodio
            td_target = reward + 0.0
        # valor esperado de la calidad de la obs que se comrara con el output de la red 
        else:
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
    def replay_experience(self, batch_size = None): # ya no va a pasar por parametro
        
        # vamos a gtraer   desde el documento de json
        batch_size = batch_size if batch_size is not self.params["replay_batch_size"]
        experience_batch = self.memory.sample(batch_size)
        # nuevometodo de aprendizaje
        self.learn_from_batch_experience(experience_batch)
        self.training_steps_completed += 1 # acutalizamos el contador cuantas hemos completado
    

    
   

    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fframento de recuerdos anteriores
        :return:
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)/255.0 # entre 0 y 1 para q sea mas sensillo de aprender 
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        if self.params["clip_rewards"]:
            reward_batch = np.sign(reward_batch) # me devuelve el signo

        next_obs_batch = np.array(batch_xp.next_obs)/255.0
        done_batch = np.array(batch_xp.done)
        
        if self.params["use_target_network"]:
            if self.step_num % self.params["target_network_update_frequency"] == 0: # si es multiplo de 200
                self.Q_target.load_state_dict(self.Q.state_dict()) #  cargaremos el dictionario habitual ya se guardaran los 200 iteraciones
            td_target = reward_batch + ~done_batch *\
                np.tile(self.gamma, len(next_obs_batch)) *\
                self.Q_target(next_obs_batch).max(1)[0].data
        else:
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

    def save(self, env_name):
        file_name = self.params["save_dir"] + "DQL_"+env_name+".ptm"
        agent_state = {"Q": self.Q.state_dict(),
                      "best_reward_mean": self.best_mean_reward,
                      "best_reward": self.best_reward
                      }
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en : ", file_name)

    def load(self, env_name):
        file_name = self.params["load_dir"]+"DQL_"+env_name+".ptm"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        self.best_mean_reward = agent_state["best_reward"]
        print("Cargado del modelo Q desde ", file_name, "que hasta el momento tiene una mejor recompensa media de: ", self.best_mean_reward, 
              "y una recompensa maxima de: ", self.best_reward)



# metodo global de aprendizaje

if __name__ == "__main__":
    environment = gym.make("LunarLander-v2")
    #environment = gym.make("CartPole-v0")
    #environment = gym.make("MountainCar-v0")
   ## environment = gym.make("BipedalWalker-v2")
    #environment = gym.make("Acrobot-v1")
    agent = SwallowQLearner(environment)
    first_episode = True
    episode_rewards = list()
    max_reward = float("-inf")
    for episode in range(MAX_NUM_EPISODE):
        obs = environment.reset() # primera observacion
        total_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            environment.render() # para verlo visual
            action = agent.get_action(obs)
            # extraigo los valores
            next_obs, reward, done, info = environment.step(action)

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
            #     environment.reset()
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
    environment.close()
