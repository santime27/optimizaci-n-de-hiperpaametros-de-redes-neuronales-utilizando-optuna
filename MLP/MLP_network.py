import os
import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, random_split

DIR = os.getcwd()

class MLPNetwork(nn.Module):
    def __init__(self,n_layers,n_neurons,activationlist,num_classes):
        """
        Args:
        n_layers (int): Numero de capas fully connected
        n_neurons (list): Lista con la cantidad de neuronas por capa
        activationlist (list): Lista con las funciones de activacion por capa
        num_classes (int): Numero de clases de salida
        """
        super().__init__() #se llaman los metodos y atributos de la clase padre (POO)
        self.flatten  = nn.Flatten()
        self.red      = nn.Sequential()
        in_features   = 28 * 28

        #se realiza un ciclo for para agregar las capas, segun lo indicado
        for i in range(n_layers):
            #se agrega a las capas la cantidad de neuronas de salida
            self.red.append(torch.nn.Linear(in_features, n_neurons[i]))
            if activationlist[i] == 'relu':
                self.red.append(nn.ReLU())
            elif activationlist[i] == 'tanh':
                self.red.append(nn.Tanh()) 
            else:
                self.red.append(nn.Sigmoid()) 
            in_features = n_neurons[i]
        
        #se agrega la ultima capa de salida
        self.red.append(torch.nn.Linear(in_features, num_classes))

    def forward(self,x):
        """Funcion para realizar la propagacion hacia adelante
        
        Args:
        x (torch.tensor): Caracteristicas de entrada
        
        Returns:
        logits (torch.tensor): Caracteristicas de salida"""
        
        x       = self.flatten(x)
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

    # Cargar el conjunto de datos de entrenamiento completo
    full_train_dataset = datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor())

    # Determinar el tamaño del conjunto de entrenamiento (2000 imágenes)
    train_size  = 2000
    val_size    = 300
    test_size   = 100

    # Dividir el conjunto de datos original en un conjunto de entrenamiento de 2000 imágenes
    train_dataset, _ = random_split(full_train_dataset, [train_size, len(full_train_dataset) - train_size])

    # Cargar el conjunto de datos de validación y prueba a partir del conjunto de validación original
    full_validation_dataset = datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor())

    # Determinar el tamaño de los conjuntos de validación (300 imágenes) y prueba (100 imágenes

    # Dividir el conjunto de validación original en validación (300) y prueba (100)
    validation_dataset, test_dataset,_ = random_split(full_validation_dataset, [val_size, test_size,len(full_validation_dataset) - val_size - test_size])

    # Crear los DataLoaders para entrenamiento, validación y prueba
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader, test_loader, len(full_train_dataset.classes)

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
def test(dataloader, model, loss_fn, device):
    """Función para realizar la validación de la red
    
    Args:
    dataloader (torch.utils.data.DataLoader): Dataloader con los datos de validación
    model (torch.nn.Module): Modelo de la red
    loss_fn (torch.nn.Module): Función de pérdida
    device (torch.device): Dispositivo donde se realiza el entrenamiento
    
    Returns:
    accuracy (float): Porcentaje de aciertos de la red
    test_loss (float): Pérdida promedio del conjunto de validación
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    # Desactiva el cálculo de gradientes para ahorrar memoria y mejorar el rendimiento
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Realiza las predicciones
            pred = model(X)
            
            # Calcula la pérdida acumulada
            test_loss += loss_fn(pred, y).item()
            
            # Calcula el número de predicciones correctas
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Promedio de la pérdida en el conjunto de validación
    test_loss /= num_batches
    
    # Precisión total
    accuracy = (correct / size) * 100

    return accuracy, test_loss