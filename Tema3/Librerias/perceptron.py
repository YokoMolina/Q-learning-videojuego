
import torch

#single layer PERCEPTRON 
# neurona con una sola capa para aproximar funciones
# hereda las funciones de la libreria torch.nn
class SLP(torch.nn.Module):
    def __init__(self,input_shape,output_shape, device = torch.device("cpu")):
        # input_shape: tamaño o forma de los datos de entrada
        #output_shape: "" de los datos de salida
        # device: el dispositivo (cpu o cuda) que la 
        # SLP debe utilizar para almacenar los inputs
        # a cada iteracion # cpu o memoria 
        # asi xq se esta heredando
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40 # 40 unidades en la capa

        # se hace la combinacion lienal para pasar a las nueronas
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape) 
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)

    def forward(self, x) : # voy a activar las neuronas
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x))# funcion de activacion RELU max{0,x}
        x = self.out(x)
        return(x)

