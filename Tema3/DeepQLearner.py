
# life esta viviendo en el entorno ... entrenando
# game cuando ya esta jugando con todo lo que aprendió --test
# se puede ver los resultados en tensorboard y se ñpuede comparar 
# los resultados del entrenamosnto con los de --test que ya es poner en prectica todo lo aprendido



# para renderizar --render 
import torch
import numpy as np
import random
import gym
from gym.spaces import Box
from datetime import datetime

from argparse import ArgumentParser # para poner argumentos en consola

#_____________________-
from Librerias.perceptron import SLP
# utilidades que se pueden programar y llamar 
# como funciones de decrecimientoTe
from utils.decay_schedule import LinearDecaySchedule
from utils.experience_memory import ExperienceMemory, Experience
#__________________________________-
from  Librerias.cnn import CNN
from utils.params_manager import ParamsManager
#_________________________-

import environments.atari as Atari
import environments.utils as env_utils

#___________________________________-
from tensorboardX import SummaryWriter # nos ayuda a guardar los valores de entrenamiento y poder analizarlos


# PongNoFrameskip-v4

# parseador de Argumentos
# con el argumento test ya sabe jugar el juego 
args = ArgumentParser("DeepQLearning")
args.add_argument("--params-file", help = "Path del fichero JSOn de parámetros. El valor por defecto es parameters.json",
                  default = "parameters.json", metavar = "PFILE")
args.add_argument("--env", help = " ENTORNO DE ID de Atari disponible en Open AI Gym. eL VALOR POR DEFECTI SERA sEAQUESnOfrMAESkip-v4",
                  default = "SeaquestNoFrameskip-v4", metavar= "ENV")
args.add_argument("--gpu-id", help = "ID de la GPU a utilizar, por defecto 0", default=0,
                  type= int, metavar="GPU_ID")
args.add_argument("--test", help = "Modo de testing para jugar sin aprender. Por defecto esta desactivado",
                  action = "store_true", default = False)
args.add_argument("--render", help = " Renderiza el entorno en pantalla. Desactivado por defecto", action = "store_true", default= False)
args.add_argument("--record", help = "Almacena videos y estados de la performance del agente",
                  action= "store_true", default = False)
args.add_argument("--output-dir", help = "Directorio para almacenar los outputs. Por defecto = ./modelo/results",
                  default= "./modelo/results")
args = args.parse_args()


# Parametrso globales
#manager = ParamsManager("parameters.json")
manager = ParamsManager(args.params_file)
#ficheros de la configuracion de las ejecuciones
summary_filename_prefix = manager.get_agent_params()["summary_filename_prefix"]
summary_filename = summary_filename_prefix + args.env + datetime.now().strftime("%y-%m-%d-%H-%M")

# Summary witer de TensorBoardX
writer = SummaryWriter(summary_filename) # donde se guardan los resultados 

manager.export_agent_params(summary_filename + "/" + "agent_params.json")
manager.export_environment_params(summary_filename + "/" + "environment_params.json")




# contador global de ejecuciones
global_step_num = 0
# habilidatar entrenamiento po gpu o cpu
use_cuda = manager.get_agent_params()["use_cuda"]
device = torch.device("cuda:"+ str(args.gpu_id)) if torch.cuda.is_available() and use_cuda else "cpu" 
# habilitar la semilla aleatoria para poder reposucir  el experiemnto despues
seed = manager.get_agent_params()["seed"]                                      
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)
                                          





class DeepQLearner (object): # ya nos le vamos a pasar el environment
    def __init__(self, obs_shape, action_shape, params):
        
        self.device = device
        self.params = params
        self.gamma = self.params["gamma"]
        self.learning_rate = self.params["learning_rate"]
        self.best_reward_mean = -float("inf")
        self.best_reward = -float("inf")
        self.best_mean_reward = -float("inf")  # Añadido: Inicialización de best_mean_reward
        self.training_steps_completed = 0
        self.action_shape = action_shape

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
                                                 max_steps = self.params["epsilon_decay_final_step"])
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
        writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)
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
        writer.add_scalar("DQL_/td_error", td_error.mean(), self.step_num)
        
        # optimiza los pesos de la red con Adam
        self.Q_optimazer.step()

    # vamos a incluir la experiencia/ memoria
    def replay_experience(self, batch_size = None): # ya no va a pasar por parametro
        
        # vamos a gtraer   desde el documento de json
        batch_size = batch_size if batch_size is not None else self.params["replay_batch_size"]
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
        
        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_frequency'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch *\
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        torch.max(self.Q_target(next_obs_batch),1)[0].data.tolist()
            td_target = torch.from_numpy(td_target)

        else: 
            td_target = reward_batch + ~done_batch * \
                        np.tile(self.gamma, len(next_obs_batch)) * \
                        torch.max(self.Q(next_obs_batch).detach(),1)[0].data.tolist()
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
        model_save_name = f'DQL_{env_name}.ptm'
        save_path = f"D:/Q-learning-videojuego/Tema3/modelo/{model_save_name}" 
        agent_state = {
            "Q": self.Q.state_dict(),
            "best_mean_reward": self.best_mean_reward,
            "best_reward": self.best_reward
        }
        torch.save(agent_state, save_path)
        print("Estado del agente guardado en:", save_path)

    

    def load(self, env_name):
        model_load_name = f'DQL_{env_name}.ptm'
        load_path = f"D:/Q-learning-videojuego/Tema3/modelo/{model_load_name}"
        try:
            agent_state = torch.load(load_path, map_location=torch.device('cpu'))
        except FileNotFoundError:
            print(f"ERROR: No se encontró ningún modelo entrenado para el entorno '{env_name}'. Comenzando desde cero.")
            return

        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(device)
        self.best_mean_reward = agent_state.get("best_mean_reward", 0)
        self.best_reward = agent_state.get("best_reward", 0)
        print("Modelo Q cargado desde", load_path,
            "con mejor recompensa media:", self.best_mean_reward,
            "y mejor recompensa:", self.best_reward)



# metodo global de aprendizaje

if __name__ == "__main__":
    # ahora vamos
    env_conf = manager.get_environment_params()
    env_conf["env_name"] = args.env
    #environment = gym.make("LunarLander-v2")
    #environment = gym.make("CartPole-v0")
    #environment = gym.make("MountainCar-v0")
   ## environment = gym.make("BipedalWalker-v2")
    #environment = gym.make("Acrobot-v1")
    if args.test:
        env_conf["episodic_life"] = False # ayuda a reportar la media 
        # en vez de hacer de lor bida
        # se imprimen para cada vida de la partida
    reward_type = "LIFE" if env_conf["episodic_life"] else "GAME"

    custom_region_available = False
    for key, value in env_conf["useful_region"].items():
        if key in args.env:
            env_conf["useful_region"] = value
            custom_region_available = True
            break
    if custom_region_available is not True: # si no hay region personalizada
        env_conf["useful_region"] = env_conf["useful_region"]["Default"]
        #por defecto
        print("Configuración a utilizar", env_conf)

    atari_env = False
    for game in Atari.get_games_list():
        if game.replace("_","") in args.env.lower(): # si hay en la lista el q el usuario puso
            atari_env = True
    if atari_env:
        environment = Atari.make_env(args.env, env_conf)
    else:
        environment = env_utils.ResizeReshapeFrames(gym.make(args.env))


    obs_shape = environment.observation_space.shape
    action_shape =environment.action_space.n
    agent_params = manager.get_agent_params()
    agent_params["test"] = args.test
    agent_params["clip_rewards"] = env_conf["clip_rewards"]

    agent = DeepQLearner(obs_shape, action_shape, agent_params)

    episode_rewards = list()
    previous_checkpoint_mean_ep_rew = agent.best_mean_reward
    num_improved_episodes_before_checkpoint = 0
    if agent_params["load_trained_model"]:
        try:
            agent.load(env_conf["env_name"])
            previous_checkpoint_mean_ep_rew = agent.best_mean_reward
        except FileNotFoundError:
            print("ERROR: no existe ningun modelo entrenado para este entorno. Empezamos desde cero")

    episode = 0
    while global_step_num < agent_params["max_training_steps"]:
        obs = environment.reset()
        total_reward = 0.0
        done = False # no hayamos terminaod
        step = 0
        while not done:
            if env_conf["render"] or args.render:
                environment.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            total_reward += reward
            step += 1
            global_step_num += 1

            if done is True:
                episode += 1
                episode_rewards.append(total_reward)
                if total_reward > agent.best_reward:
                    agent.best_reward = total_reward
                if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew:
                     num_improved_episodes_before_checkpoint += 1 # se detecto una mejor

                if num_improved_episodes_before_checkpoint >= agent_params["save_freq"]:
                    # si subimos de 50  se almacenan
                    previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean_reward = np.mean(episode_rewards)
                    agent.save(env_conf["env_name"])
                    num_improved_episodes_before_checkpoint = 0
                print("\n Episodio #{} finalizado con {} iteraciones. Con {} estados: recompensa = {}, recompensa media = {:.2}, mejor recompensa = {}".format(episode, step+1, reward_type, total_reward, np.mean(episode_rewards), agent.best_reward))

                writer.add_scalar("main/ep_total_reward", total_reward, global_step_num)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
                writer.add_scalar("main/max_ep_reward", agent.best_mean_reward, global_step_num)
                #writer.add_scalar("main/total_reward", agent.best_mean_reward, global_step_num)

                if agent.memory.get_size() >= 2*agent.params["replay_start_size"] and not args.test:
                    agent.replay_experience()
                break

    environment.close()
    writer.close()
