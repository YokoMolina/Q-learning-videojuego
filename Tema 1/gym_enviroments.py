from gym import envs
    
# me devuelve los nombres de los enviroments    
env_names = [env.id for env in envs.registry.all()]

for name in sorted(env_names):
    print(name)

# los que tienen ram se ejecutan en la memoria y no en el disco

# deterministic, son intervalos deterministas de 4 frames, con resultado periódico
    
# noframe_ el agente toma la acción y en el mismo fráme obtiene el estado nuevo

