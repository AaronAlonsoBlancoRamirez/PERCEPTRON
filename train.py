import numpy as np

# Evaluación simple del modelo en el conjunto de prueba
def evaluate(network, X, y):
    predictions = network.forward_pass(X)[-1]
    predicted_labels = np.argmax(predictions, axis=0)
    true_labels = np.argmax(y, axis=0)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy
