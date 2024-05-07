

import torch
import numpy as np
from Librerias.perceptron import SLP
# utilidades que se pueden programar y llamar 
# como funciones de decrecimiento
from utils.decay_schedule import LinearDecaySchedule
import random
import gym
from gym.spaces import Box

MAX_NUM_EPISODE = 100000
STEPS_PER_EPISODE = 300


class SwallowQLearner (object):
    def __init__(self, environment, learning_rate = 0.005, gamma = 0.98):
        #inicialicemos los valores del objeto self
        self.learning_rate = learning_rate # podemos guardar el dato
        self.obs_shape = environment.observation_space.shape
        
        self.action_shape = environment.action_space.n

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



    def get_action(self, obs):
        # retornar la politica dependiendo de la observacion
        return self.polity(obs)
    
    # la politica 
    def epsilon_greedy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num): # funcion de decrecimiento que le paso el numero de iteracion
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device("cpu")).numpy())
        return action

    def learn(self, obs, action, reward, next_obs):
        
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        # diferencial temporal
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        # para optimizar los valores de los pesos hacia atras 
        self.Q_optimazer.zero_grad()
        td_error.backward() # hace el proceso hacia atras
        # optimiza los pesos de la red
        self.Q_optimazer.step()


if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
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
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

#ctrls+c
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
                break
    environment.close()
