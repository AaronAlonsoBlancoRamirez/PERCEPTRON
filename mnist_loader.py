import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    """
    Carga y prepara el dataset MNIST.
    :return: Tupla (X_train, y_train, X_test, y_test) con los datos normalizados.
    """
    # Cargar dataset MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizar los datos (escalar valores de píxeles entre 0 y 1)
    X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255

    # Convertir etiquetas a vectores binarios (one-hot encoding)
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train.T, y_train.T, X_test.T, y_test.T