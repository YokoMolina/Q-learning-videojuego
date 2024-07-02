
import torch

class CNN(torch.nn.Module):
    # una red neuronal convolucional que tomará decisiones según,
    # los píxeles de la imagen

    def __init__(self, input_shape, output_shape, device = "cpu"):
        # input:shape: es la dimension de la imagen que supondremos 
        # viene rescalada a Cx84x84 
        # output shape: dimension de la salida
        # device dispositivo (cpu o gpu) donde la cnn debe
        # almacenar los valores de cada iteracion

        # input_shape 84x84 supuesto
        super(CNN, self).__init__()
        self.device = device
        # vamos a ir filtrando la informacon con los kernels
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 64, kernel_size = 4, stride = 2, padding = 1),
            torch.nn.ReLU()
            
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size = 4, stride =2, padding = 0),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 0),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Linear(18*18*32, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float.to(self.device)
        x = self.layer1
        x = self.layer2
        x = self.layer3
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x 

