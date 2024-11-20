from mnist_loader import load_data
import tensorflow as tf
from network import Network
from train import evaluate
import matplotlib.pyplot as plt
from convert_weights import convert_weights_format
import numpy as np 
from PIL import Image
import os

def save_sample_images(X, y, path='images'):
    """
    Guardar imágenes de los dígitos del 0 al 9 desde el conjunto de prueba.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # Si `y` está en formato one-hot, convertirlo a un array simple con etiquetas
    if y.ndim > 1:
        y = y.argmax(axis=0)

    for digit in range(10):
        # Encontrar la primera ocurrencia del dígito en y
        indices = np.where(y == digit)[0]
        if len(indices) == 0:
            print(f"No se encontró el dígito {digit} en el conjunto de prueba.")
            continue

        index = indices[0]

        # Obtener la imagen correspondiente (teniendo en cuenta la forma correcta de X_test)
        sample = X[:, index].reshape(28, 28)

        # Guardar la imagen sin márgenes (axes) para asegurar el tamaño correcto de 28x28
        fig, ax = plt.subplots(figsize=(1, 1), dpi=28)
        ax.imshow(sample, cmap='gray')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'{path}/digit_{digit}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# Define la función para predecir desde imágenes guardadas
def predict_from_images(network, images_folder='images'):
    """
    Función para predecir los dígitos de las imágenes guardadas en la carpeta 'images'.
    """
    for digit in range(10):
        # Construir la ruta de cada archivo de imagen
        image_path = os.path.join(images_folder, f'digit_{digit}.png')

        # Comprobar si el archivo existe
        if not os.path.exists(image_path):
            print(f"Imagen para el dígito {digit} no encontrada.")
            continue

        # Cargar la imagen usando PIL y convertirla a escala de grises
        img = Image.open(image_path).convert('L')

        # Convertir la imagen a un arreglo de numpy y redimensionar a 28x28
        img_array = np.array(img)

        # Normalizar los valores (de 0-255 a 0-1)
        img_array = img_array / 255.0

        # Aplanar la imagen para que sea compatible con la entrada de la red (784, 1)
        img_flattened = img_array.flatten().reshape(-1, 1)

        # Hacer la predicción con la red neuronal
        prediction = network.forward_pass(img_flattened)[-1].argmax()

        # Mostrar la imagen junto con la predicción
        plt.imshow(img_array, cmap='gray')
        plt.title(f"Etiqueta verdadera: {digit}, Predicción: {prediction}")
        plt.axis('off')
        plt.show()

def main():
    # Configuración de hiperparámetros
    layer_sizes = [784, 256, 128, 64, 10]  # Capas de la red: entrada, dos ocultas más pequeñas, salida
    learning_rate = 0.001
    epochs = 100  # Reducir a 1 época para un entrenamiento rápido

    # Cargar el dataset MNIST
    print("Cargando dataset MNIST...")
    X_train, y_train, X_test, y_test = load_data()

    # Inicializar la red neuronal
    network = Network(layer_sizes)

    # Entrenar la red
    print("Entrenando la red neuronal...")
    network.train(X_train, y_train, epochs, learning_rate, batch_size=64)

    # Guardar los pesos del modelo entrenado
    network.save_weights('weights.txt')

    # Convertir el archivo a formato compatible con la interfaz gráfica
    input_file = 'weights.txt'
    output_file = 'weights_converted.txt'
    convert_weights_format(input_file, output_file)

    # Evaluar el modelo en el conjunto de prueba
    print("Evaluando el modelo en el conjunto de prueba...")
    accuracy = evaluate(network, X_test, y_test)
    print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

    # Guardar una imagen de cada dígito del conjunto de prueba
    save_sample_images(X_test, y_test)

    # Predecir y visualizar cada imagen guardada
    predict_from_images(network)

if __name__ == "__main__":
    main()
