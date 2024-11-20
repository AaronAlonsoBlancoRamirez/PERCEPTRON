def convert_weights_format(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Variables para almacenar los valores
    flattened_values = []

    # Recorrer las líneas y extraer los valores
    for line in lines:
        if "Capas" in line or "Sesgos" in line:
            # Ignorar las líneas que contienen etiquetas como "Capas" o "Sesgos"
            continue
        else:
            # Dividir la línea por espacios y agregar los valores a la lista
            values = line.split()
            flattened_values.extend(values)

    # Cambiar puntos por comas en cada valor
    flattened_values = [value.replace('.', ',') for value in flattened_values]

    # Escribir los valores en el archivo de salida
    with open(output_file, 'w') as outfile:
        outfile.write(' '.join(flattened_values))

# Uso del script
input_file = 'weights.txt'  # Archivo de entrada con el formato actual
output_file = 'weights_converted.txt'  # Archivo de salida con el nuevo formato
convert_weights_format(input_file, output_file)

#print("Conversión completada. Archivo guardado como 'weights_converted.txt'")
