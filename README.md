# optimización de hiperpaametros de redes neuronales utilizando optuna

Optuna es un framework de optimización de hiperparámetros automatizado, diseñado para encontrar las configuraciones óptimas de hiperparámetros en modelos de machine learning y deep learning de forma eficiente. Optuna permite definir dinámicamente el espacio de búsqueda de hiperparámetros y utiliza algoritmos avanzados como el Tree-structured Parzen Estimator (TPE), un método de optimización bayesiana que modela la relación entre los hiperparámetros y el rendimiento del modelo, explorando de manera más profunda las configuraciones prometedoras. Además, Optuna incorpora el Asynchronous Successive Halving Algorithm (ASHA), que permite descartar rápidamente las configuraciones menos prometedoras durante el entrenamiento, optimizando así el uso de los recursos computacionales. También admite búsquedas aleatorias y algoritmos como el Simulated Annealing para la optimización global. Optuna es ideal para escenarios distribuidos, permitiendo realizar optimizaciones en varios nodos de computación y ofreciendo integración con herramientas como Kubernetes y Docker. En este repositorio se utiliza Optuna para optimizar los hiperparámetros en redes neuronales como redes completamente conectadas (MLP), redes convolucionales (CNN) y redes neuronales de grafos (GNN), aplicadas a tareas de clasificación. Para más detalles sobre Optuna, visita [Optuna en GitHub](https://github.com/optuna/optuna).

# Descripción del proyecto

Para este proyecto, se realizó la optimización de hiperparámetros en arquitecturas de redes neuronales profundas como CNN, MLP y GNN, utilizando el framework Optuna. El objetivo fue encontrar las configuraciones óptimas que maximizaran el rendimiento de cada red en tareas de clasificación.

# Estructura del Repositorio

Este repositorio está dividido en tres carpetas principales, que corresponden a las arquitecturas de redes neuronales que serán optimizadas:

- CNN/
    - CNN_network.py
    - optuna_cnn_train.ipynb
- GNN/
    - GCN_network.py
    - optuna_gcn_train.ipynb
- MLP/
    - MLP_network.py
    - optuna_mlp_train.ipynb
- requirements.txt


Cada carpeta de la arquitectura de red neuronal contiene una clase que define la estructura de la red neuronal, partiendo de las entradas específicas, ademas, tambien contiene
Un notebook de Jupyter que contiene el procedimiento completo de optimización de hiperparámetros utilizando Optuna para cada red neuronal.


Este repositorio incluye un archivo requirements.txt que lista las bibliotecas necesarias para ejecutar el proyecto, facilitando la instalación de dependencias.