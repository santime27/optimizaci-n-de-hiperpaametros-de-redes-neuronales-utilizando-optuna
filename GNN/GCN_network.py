from   torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn
import torch

class GCNNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, layers):
        """
        Args:
        num_features (int): Número de características de entrada
        num_classes (int): Número de clases de salida
        layers (list): Lista con la cantidad de neuronas por capa
        """
        super().__init__()
        torch.manual_seed(1234567)
        self.convs  = torch.nn.ModuleList()
        in_features = num_features
        
        # Agregar capas GCN
        for i in layers:
            self.convs.append(GCNConv(in_features, i))
            in_features = i  # Actualizar in_features para la siguiente capa
        
        # Capa final
        self.convs.append(GCNConv(in_features, num_classes))

    def forward(self, x, edge_index):
        """Función para realizar la propagación hacia adelante

        Args:
        x (torch.tensor): Características de entrada
        edge_index (torch.tensor): Índices de los bordes
        
        Returns:
        x (torch.tensor): Características de salida"""
        
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Aplicar la última capa sin activación
        x = self.convs[-1](x, edge_index)
        return x

def train(data, model, loss_fn, optimizer,edge_index):
    """Función para realizar el entrenamiento de la red
    
    Args:
    data (torch_geometric.data.Data): Datos de entrenamiento
    model (torch.nn.Module): Modelo de la red
    loss_fn (torch.nn.Module): Función de pérdida
    optimizer (torch.optim): Optimizador
    edge_index (torch.tensor): Índices de los bordes
    
    Returns:
    loss (float): Pérdida de la red"""

    #se prepara el modelo para entrenar
    model.train()

    # Realizar predicciones
    pred = model(data.x,edge_index)

    # Calcular la perdida
    loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def validation(data, model,edge_index):
    """Función para realizar la validación de la red
    
    Args:
    data (torch_geometric.data.Data): Datos de validación
    model (torch.nn.Module): Modelo de la red
    edge_index (torch.tensor): Índices de los bordes
    
    Returns:
    val_acc (float): Porcentaje de aciertos de la red"""

    # se pone el modelo en modo de evaluacion
    model.eval()

    # Realizar predicciones
    pred = model(data.x,edge_index)

    #se busca la clase con mayor prediccion
    out = pred.argmax(dim=1)

    # se verifica la prediccion
    val_correct = out[data.val_mask] == data.y[data.val_mask]

    #se calculan las predicciones correctas
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
    return val_acc*100

def test(data, model,edge_index):
    """Función para realizar la validación de la red
    
    Args:
    data (torch_geometric.data.Data): Datos de validación
    model (torch.nn.Module): Modelo de la red
    edge_index (torch.tensor): Índices de los bordes
    
    Returns:
    test_acc (float): Porcentaje de aciertos de la red"""

    # se pone el modelo en modo de evaluacion
    model.eval()

    # Realizar predicciones
    pred = model(data.x,edge_index)

    #se busca la clase con mayor prediccion
    out = pred.argmax(dim=1)

    # se verifica la prediccion
    test_correct = out[data.test_mask] == data.y[data.test_mask]

    #se calculan las predicciones correctas
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc*100