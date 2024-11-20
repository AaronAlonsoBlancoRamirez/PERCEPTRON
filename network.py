import numpy as np
 
class Network:
    def __init__(self, layer_sizes):
        """
        Inicializa la red con los tamaños de cada capa y los pesos aleatorios.
        :param layer_sizes: Lista con el número de neuronas en cada capa (incluyendo entrada y salida).
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        # Inicializamos los pesos y sesgos con valores pequeños para estabilidad
        self.weights = [np.random.normal(0, 0.01, (y, x)) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.normal(0, 0.01, (y, 1)) for y in layer_sizes[1:]]
        # Lista para almacenar la pérdida en cada época
        self.losses = []

    def forward_pass(self, X):
        """
        Realiza la propagación hacia adelante.
        :param X: Entrada de la red (numpy array de tamaño [n_entradas, n_muestras]).
        :return: Activaciones de todas las capas.
        """
        activations = [X]
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activations[-1]) + b
            if i == self.num_layers - 2:  # Última capa
                activations.append(self.sigmoid(z))
            else:  # Capas ocultas
                activations.append(self.relu(z))
        return activations

    def backward_pass(self, X, y):
        """
        Realiza la retropropagación para calcular los gradientes de los pesos y los sesgos.
        :param X: Entrada de la red.
        :param y: Salida esperada.
        :return: Gradientes de los pesos y los sesgos.
        """
        # Propagación hacia adelante
        activations = self.forward_pass(X)
        # Inicializamos listas para almacenar gradientes
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        
        # Calculamos el error en la última capa
        delta = (activations[-1] - y) * self.sigmoid_derivative(activations[-1])
        d_weights[-1] = np.dot(delta, activations[-2].T)
        d_biases[-1] = np.sum(delta, axis=1, keepdims=True)
        
        # Retropropagación del error hacia capas anteriores
        for l in range(2, self.num_layers):
            if l == self.num_layers - 1:
                delta = np.dot(self.weights[-l + 1].T, delta) * self.relu_derivative(activations[-l])
            else:
                delta = np.dot(self.weights[-l + 1].T, delta) * self.sigmoid_derivative(activations[-l])
            d_weights[-l] = np.dot(delta, activations[-l - 1].T)
            d_biases[-l] = np.sum(delta, axis=1, keepdims=True)
        
        return d_weights, d_biases

    def update_weights(self, d_weights, d_biases, learning_rate):
        """
        Actualiza los pesos de la red usando los gradientes calculados.
        :param d_weights: Gradientes de los pesos.
        :param d_biases: Gradientes de los sesgos.
        :param learning_rate: Tasa de aprendizaje para el ajuste de los pesos.
        """
        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, d_weights)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, d_biases)]

    def train(self, X, y, epochs, learning_rate, batch_size=32):
        """
        Entrena la red usando el conjunto de datos proporcionado con mini-batch.
        :param X: Conjunto de datos de entrada.
        :param y: Conjunto de etiquetas esperadas.
        :param epochs: Número de épocas para el entrenamiento.
        :param learning_rate: Tasa de aprendizaje.
        :param batch_size: Tamaño del mini-batch.
        """
        n_samples = X.shape[1]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[:, indices]
            y_shuffled = y[:, indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                d_weights, d_biases = self.backward_pass(X_batch, y_batch)
                d_weights = [np.clip(dw, -1, 1) for dw in d_weights]
                d_biases = [np.clip(db, -1, 1) for db in d_biases]
                self.update_weights(d_weights, d_biases, learning_rate)

            # Calculamos el error actual (pérdida) y almacenamos
            predictions = self.forward_pass(X)[-1]
            loss = np.mean((y - predictions) ** 2)
            self.losses.append(loss)

            # Imprimir pérdida en cada época
            print(f"Epoch {epoch}, Loss: {loss}")

    def save_weights(self, file_path='weights.txt'):
        """
        Guarda los pesos y sesgos en un archivo de texto.
        :param file_path: Ruta donde se guardarán los pesos y sesgos.
        """
        with open(file_path, 'w') as f:
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                f.write(f"Capas {i}:\n")
                np.savetxt(f, w, fmt='%0.8f')
                f.write("\nSesgos:\n")
                np.savetxt(f, b, fmt='%0.8f')
                f.write("\n")
        print(f"Pesos y sesgos guardados en {file_path}.")

    def load_weights(self, file_path='weights.npy'):
        """
        Carga los pesos y sesgos desde un archivo.
        :param file_path: Ruta desde donde se cargarán los pesos y sesgos.
        """
        data = np.load(file_path, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']
        print(f"Pesos y sesgos cargados desde {file_path}.")

    def save_loss(self, file_path='loss.npy'):
        """
        Guarda la pérdida durante el entrenamiento en un archivo.
        :param file_path: Ruta donde se guardará la pérdida.
        """
        np.save(file_path, self.losses)
        print(f"Pérdida guardada en {file_path}.")

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)  # Limita el valor de z para evitar el desbordamiento
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)
