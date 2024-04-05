
# metodos/funciones:

# ___init__(self, enviroment)
# discretize(self,obs) # se discretiza los espacios para que no sean infinitos [-2,2]
# get_action(self, obs)
# learn(self, obs action, reward, next_obs)
# EPSIOLON_MIN : vamos aprendiendo mintras el incremento del aprendizaje sea superior a ese valor
# para garantizar la convergencia a fuerza bruta
# MAX_NUM_EPISODES: numero de iteraciones dispuestos a realizar
# STEPS_PER_EPISODE: numero max de pasos a realizar en cada episodio
# ALPHA: ratio de aprendizaje / podria ir variando no puede solo ser constante
# GAMMA: factor de descuento del agente
# NUM_DISCRETE_BINS: numero de divisiones en caso de diviir el espacio de estados continuo

#_________________--
# CLASS Q LEARNER
import numpy as np
import gym

EPSILON_MIN = 0.005
MAX_NUM_EPISODE = 1000
STEPS_PER_EPISODE =200
max_min_steps = MAX_NUM_EPISODE*STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_min_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DICRETE_BINS = 30

class QLearner (object):
    def __init__(self, enviroment):
        #inicialicemos los valores del objeto self
        self.obs_shape = enviroment.observation_space.shape
        self.obs_high = enviroment.observation_space.high
        self.obs_low = enviroment.observation_space.low
        self.obs_bins = NUM_DICRETE_BINS
        self.obs_width = (self.obs_high-self.obs_low)/self.obs_bins

        self.action_shape = enviroment.action_space.n
        self.Q = np.zeros((self.obs_bins+1,self.obs_bins+1, self.action_shape))
        # amtriz de 31x31x3
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0 #valor que se va incrementando si sobrepasa el epsiolon_min

    def discretize(self,obs):
        # discreticemos el espacio de observaciones
        return tuple(((obs-self.obs_low)/self.obs_width).astype(int))
    # astyope me ayuda a escoger el piso inferir de la division para saber en que division me encuentro

    def get_action(self, obs):
    # politica de fuerza bruta para minimizar el epsilon min
    # mejor accion con menor porbabilidad 1-epsion que es el vañor de equivocarse
        discrete_obs = self.discretize(obs)
        # seleccion de la acion en vase a Epsilon-greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon: # se elige un num random entre 0y1
            #con proba 1-epsilon elegimos la mejor posible
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])
            # con proba epsilon elegimos al azar
    def learn(self, obs, action, reward, next_obs):
        # implementación de la ecuacion q-learner
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
        td_error = td_target -self.Q[discrete_obs][action]
        self.Q[discrete_obs][action] += self.alpha*td_error

## Metodo pare entrenas a nuestro agente
        
def train(agent, enviroment):
        best_reward = -float("inf")
        for episode in range(MAX_NUM_EPISODE):
            done = False
            obs = enviroment.reset()
            total_reward = 0.0
            while not done:
                action = agent.get_action(obs) # accion elegida segun la ecuacion q-learning
                next_obs, reward, done, info= enviroment.step(action)
                agent.learn(obs,action,reward, next_obs)
                obs= next_obs
                total_reward += reward
            if total_reward > best_reward:
                best_reward = total_reward
            print("Episodio número {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(episode, total_reward, best_reward, agent.epsilon))
    # de todas las politicas de entrenamiento que emos obtenido
            # devolvemos la mejor de todas
        return np.argmax(agent.Q, axis=2)

# cuanto aprendio
def test(agent, enviroment, policy):
    done = False
    obs = enviroment.rest()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)] # accion que dictamila la politica que hemos entrenado
        next_obs, reward, done, info= enviroment.step(action)
       
        obs = next_obs
        total_reward += reward
    return total_reward

# almacenar  CADA ENTRENAMOIENTO

if __name__ == "__main__":
    enviroment = gym.make("MontainCar-v0")
    agent = QLearner(enviroment)
    learned_policy = train(agent, enviroment)
    monitor_path = "./monitor_output"
    enviroment = gym.wrappers.Monitor(enviroment, monitor_path, force=True)
    for _ in range(1000):
        test(agent, enviroment, learned_policy)
    enviroment.close()







