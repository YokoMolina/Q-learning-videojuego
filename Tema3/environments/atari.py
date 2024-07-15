# vamos a configurar los entornos para que pueda leer nuestra red cnn para no configurar cada entornor
# sino solo tomar cualquier juego y configurarlo desde este script
import gym
import atari_py
import numpy as np
import random 
import cv2
from collections import deque
from gym.spaces.box import Box

def get_games_list():
   return atari_py.list_games()

def make_env(env_id, env_conf):
    env = gym.make(env_id)
    if "NoFrameskip" in env_id:
        assert "NoFrameskip" in env.spec.id
        env = NoopResetEnv(env, noop_max = 30) # el estado inicial toma valores aleatorias para las priemras permite la reproductividad
        env = MaxAndSkipEnv(env, skip = env_conf["skip_rate"]) # repite una acion un numero de veces 
    if env_conf["episodic_life"]:
        env = EpisodicLifeEnv(env) # marca el fin de vida cuando se acab el episodio
    try:
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)# sirve para s
            #disparar una accion en el reset
    except AttributeError:
        pass
    
    env = AtariRescale(env, env_conf["useful_region"])

    if env_conf["normalize_observation"]:
        env = NormalizedEnv(env)
        # normalizamos las imagenes con los valores, estandarizamos con una campana de gauss
    
    env = FrameStack(env, env_conf["num_frames_to_stack"])

    if env_conf["clip_rewards"]:
        env = ClipReward(env)
        # sirve para evitar valores muy positivos o negativos 
    return env


# ORIGINAL DE ATARI A 84X84
def process_frames_84(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"]+160, :160] # matriz con el tamaño que necesito
    frame = frame.mean(2) # TRANSFORMAMOS A ESCALA DE GRISES
    frame = frame.astype(np.float32) # reduce pues un float se guarda en 1 bit 
    frame *= 1.0/255.0 # ayuda a tener un rango continuo y no discretizado
    frame = cv2.resize(frame, (84, conf["dimension2"]))
    frame = cv2.resize(frame, (84,84))
    frame = np.reshape(frame, (1,84,84)) # escalamos a 84,84,1) con un color escala de grises
    return frame 

# ATARI TIENE 210X160X3

class AtariRescale(gym.ObservationWrapper): # clase para rescalar las imagenes
    # reducimos la ram que necesitamos, son 2 experiencias en la memoria xq es la atual y la siguiete
    # los bits se duplican a almacenar

    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0, 255, [1, 84, 84], dtype=np.uint8)
        self.conf = env_conf
    
    def observation(self, observation):
        return process_frames_84(observation, self.conf)
    


class NormalizedEnv(gym.ObservationWrapper): # vamos a normalizar los datos para que me quede una campana de gauss media 0  varianza 1}
    def __init__(self, env = None):
        gym.ObservationWrapper.__init__(self, env)
        self.mean = 0
        self.std = 0
        self.alpha = 0.999
        self.num_steps = 0
    
    def observation(self, observation):
        self.num_steps += 1
        # voy a ponderar los valores para que se de mas peso a lo valores recientes 
        self.mean = self.mean * self.alpha + observation.mean() * (1-self.alpha)
        self.std = self.std *self.alpha + observation.std() * (1-self.alpha)
        # aqui ya los hago insesgados cosntruyendo un estadistico insesgado desde el sesgado
        unblased_mean = self.mean / (1-pow(self.alpha, self.num_steps))
        unblased_std = self.std /(1-pow(self.alpha, self.num_steps))
        return (observation - unblased_mean)/(unblased_std + 1e-8)

class ClipReward(gym.RewardWrapper):
    # en los diferentes juegos existen diferentes recompensas pueden ser negativas o positivas o cero
    # vamos a recortar las recompensas dependiendo de la recopensa recibida por el entorno
    # para q no sean muy diferentes entre entornos 
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
    
    def reward(self, reward):
        return np.sign(reward) # no es tan bueno dependiendo del entorno pues esta perdiendo valor en cierto sentido
    





class NoopResetEnv(gym.Wrapper): #vamos a resetar a un estado aleatorio para q el agente
    # no aprenda el estado inicial, evita memorizar el inicio del juego
    def __init__(self, env, noop_max = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"


    def reset(self):
        self.env.reset()  # Corregido: Cambiado de self.env.rest() a self.env.reset()
        noops = random.randrange(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            observation, _, done, _ = self.env.step(self.noop_action)
        return observation
    
    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper): # pulsara el boton fire para iniciar la partida
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        return obs
    
    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    # en el caso de que se pierda una vida no se resetee el entorno solo cuando se acabe el episodio
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.has_really_died = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.has_really_died = False
        lives = info["ale.lives"]
        if lives < self.lives and lives > 0:
            done = True 
            self.has_really_died = True
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self):
        if self.has_really_died is False:
            obs = self.env.reset()
        else:
            obs, _, _, info = self.env.step(0) # en el info obtengo las vidas disponibles
            self.lives = info["ale.lives"]
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip  # Esta línea se agregó para almacenar el parámetro skip como un atributo

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):  # Esta línea se actualizó para utilizar self._skip
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

            


class FrameStack(gym.Wrapper):
    # vamos a apilar los ultimos k frames  para ser procesados de golpe
    # ayuda a la eficiencia en ram
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen = k)
        shape = env.observation_space.shape
        self.observation_space = Box(low= 0, high= 255, shape = (shape[0]* k, shape[1], shape[2]), dtype= np.uint8) # de 8 bits

        # para cada canal de color almaceno k elementos 

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs) # inicializamos con k copias
        return self.get_obs()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.get_obs(), reward, done, info
    
    def get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames)) # pra q sea menos cosntoso compitacionalmente
    


class LazyFrames(object):
    def __init__(self, frames):
        self.frames = frames
        self.out = None
    
    def _force(self):
        if self.out is None:  # Corrección aquí: uso de 'is None' para comparar con None
            self.out = np.concatenate(self.frames, axis=0)
            self.frames = None
        return self.out
    
    def _array_(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]
    

if __name__ == "__main__":
    games_list = get_games_list()
    print("Lista de juegos disponibles en Atari:")
    for game in games_list:
        print(game)