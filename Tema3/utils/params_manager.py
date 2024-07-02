

import json

class ParamsManager(object):
    def __init__(self, params_file):
        #cargaremos los parametrso desde el json 
        self.params = json.load(open(params_file, "r")) # r solo lectura puede ser w 
    
    def get_params(self):
        # solo devolvemos los parametros
        return self.params
    
    def get_agent_params(self):
        return self.params["agent"]
    
    def get_environment_params(self):
        return self.params["environment"]
    
    def update_agent_params(self, **kwargs): # le estoy pasando un puntero
        for key, value in kwargs.items():
            if key in self.get_agent_params().keys():
                self.params["agent"][key] = value
    # actualiza los valores en el diccionario

    def export_agent_params(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.params["agent"], f, indent = 4, separators = (",", ":"), sort_keys = True)
            f.write("\n")


    def export_environment_params(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.params["environment"], f, indent = 4, separators = (",", ":"), sort_keys = True)
            f.write("\n")

if __name__== "__main__":
    print("Porbando ")
    param_file = "parameters.json"
    manager = ParamsManager(param_file)

    agent_params = manager.get_agent_params()
    print("parametrso del agente: ")
    for key, value in agent_params.items():
        print(key, ":", value)

    
    env_params = manager.get_environment_params()
    print("parametrso del envi: ")
    for key, value in env_params.items():
        print(key, ":", value)
    # vamos a cambiar los valores en el json
    manager.update_agent_params(learning_rate = 0.01, gamma = 0.92)
    agent_params_update = manager.get_agent_params()
    print("los actualizados")
    for key, value in agent_params_update.items():
        print(key, ":", value)

    print(" fin de la prueba")