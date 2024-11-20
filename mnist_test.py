from mnist_loader import load_data

X_train, y_train, X_test, y_test = load_data()

# Imprimir algunas formas para asegurarse de que los datos se cargaron correctamente
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
