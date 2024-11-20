import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(file_path='loss.npy'):
    """
    Carga la pérdida guardada durante el entrenamiento y la grafica.
    """
    losses = np.load(file_path)
    plt.plot(losses)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la Pérdida Durante el Entrenamiento')
    plt.show()
