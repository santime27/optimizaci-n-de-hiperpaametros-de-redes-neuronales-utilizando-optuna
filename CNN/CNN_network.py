import os
import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, random_split

DIR = os.getcwd()

#se crea el objeto de red neuronal, eredando de la clase nn.module
class CNNNetwork(nn.Module):
    def __init__(self,n_conv_layers,num_filters,n_fully_layers,num_neurons):
        """
        Args:
        n_conv_layers (int): Numero de capas convolucionales
        num_filters (list): Lista con la cantidad de filtros por capa
        n_fully_layers (int): Numero de capas fully connected
        num_neurons (list): Lista con la cantidad de neuronas por capa
        """

        super().__init__()
        self.red        = nn.Sequential()
        in_channels     = 1
        input_size      = 28                                                                    # Input image size (28 pixels)
        kernel_size     = 3 

        #se realiza un ciclo for para agregar las capas convoluciones, ademas de los filtros por capa
        for i in range(n_conv_layers):

            #se agrega a las capas convolucionales la cantidad de filtros
            self.red.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_size, padding=1))
            self.red.append(torch.nn.ReLU())
            self.red.append(nn.MaxPool2d(kernel_size=2)) 

            in_channels     = num_filters[i]
            input_size      = input_size // 2

        # Aplanar la salida de las capas convolucionales
        self.red.append(nn.Flatten())
        
        in_features = num_filters[-1] * (input_size ** 2)

        #se realiza ciclo for para agregar las capa fully connected,ademas de las neuronas por capa
        for i in range(n_fully_layers):
            self.red.append(nn.Linear(in_features, num_neurons[i]))
            self.red.append(nn.ReLU())
            in_features = num_neurons[i]

        #capa de salida
        self.red.append(nn.Linear(in_features, 10)) 

    def forward(self,x):
        """Funcion para realizar la propagacion hacia adelante
        
        Args:
        x (torch.tensor): Caracteristicas de entrada
        
        Returns:
        logits (torch.tensor): Caracteristicas de salida"""
        
        logits  = self.red(x)
        return logits

#se crea funcion para cargar los datos
def get_mnist(batch_size):
    """Funcion para cargar los datos de entrenamiento y validacion
    
    Args:   
    batch_size (int): tamaño del batch

    Returns:
    train_loader (torch.utils.data.DataLoader): Dataloader con los datos de entrenamiento
    validation_loader (torch.utils.data.DataLoader): Dataloader con los datos de validacion
    test_loader (torch.utils.data.DataLoader): Dataloader con los datos de prueba
    """

    full_dataset = datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor())

    # Determina los tamaños para el conjunto de entrenamiento y prueba
    test_size  = int(0.1 * len(full_dataset))  # 10% para prueba
    train_size = len(full_dataset) - test_size  # 90% para entrenamiento

    # Divide el conjunto de datos
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Crea los DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Crea el DataLoader para el conjunto de validación
    validation_loader = DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    return train_loader,validation_loader, test_loader

#se crea una funcion para realizar entrenamiento
def train(dataloader, model, loss_fn, optimizer,device):
    """Funcion para realizar el entrenamiento de la red
    
    Args:
    dataloader (torch.utils.data.DataLoader): Dataloader con los datos de entrenamiento
    model (torch.nn.Module): Modelo de la red
    loss_fn (torch.nn.Module): Funcion de perdida
    optimizer (torch.optim): Optimizador
    
    Returns:
    loss_avg (float): Perdida promedio del entrenamiento"""

    loss_avg = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # se realizan las predicciones
        pred = model(X)

        # se calcula la perdida
        loss = loss_fn(pred, y)
        loss_avg += loss.item()

        # se realiza la propagacion hacia atras
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss_avg /= len(dataloader)
    return loss_avg
    
#se crea una funcion para realizar la validacion
def test(dataloader, model, loss_fn,device):
    """Funcion para realizar la validacion de la red
    
    Args:
    dataloader (torch.utils.data.DataLoader): Dataloader con los datos de validacion
    model (torch.nn.Module): Modelo de la red
    loss_fn (torch.nn.Module): Funcion de perdida
    device (torch.device): Dispositivo donde se realiza el entrenamiento
    
    Returns:
    accuracy (float): Porcentaje de aciertos de la red
    test_loss (float): Perdida promedio de la red"""

    size        = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss   /= num_batches
    correct     /= size
    accuracy     = 100*correct
    return accuracy,test_loss