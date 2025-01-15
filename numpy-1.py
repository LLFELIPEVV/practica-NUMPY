import numpy as np

# Creando un array en Numpy
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# Imprimiendo la versión de Numpy
print(f"Versión: {np.__version__}")

# Verificando el tipo del array
print(f"Tipo: {type(arr)}")

# Puedes pasar listas, tuplas u otros tipos de secuencias al método array de NumPy,
# y se convertirán en un objeto ndarray (N-dimensional array).
arr = np.array((1, 2, 3, 4, 5))  # Creando un array a partir de una tupla
print(arr)  # Imprimiendo el array

# Matrices 0-D: también conocidas como escalares, son matrices de cero dimensiones.
# Un escalar es un solo número, sin dimensiones adicionales.
arr = np.array(42)
print(f"0-D: {arr}")
print(f"Dimensiones: {arr.ndim}")  # Número de dimensiones

# Matrices 1-D: también conocidas como vectores, son matrices de una sola dimensión.
# Un vector es una secuencia de elementos que tiene una sola dimensión.
arr = np.array([1, 2, 3, 4, 5])
print(f"1-D: {arr}")
print(f"Dimensiones: {arr.ndim}")  # Número de dimensiones

# Matrices 2-D: también conocidas como matrices bidimensionales, son arreglos que contienen matrices 1-D como sus elementos.
# Estas matrices son similares a las tablas de datos con filas y columnas.
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2-D: {arr}")
print(f"Dimensiones: {arr.ndim}")  # Número de dimensiones

# Matrices 3-D: también conocidas como arreglos tridimensionales, son matrices que contienen matrices 2-D como sus elementos.
# Estas matrices son útiles para representar datos con múltiples capas o niveles, como una colección de imágenes en escala de grises,
# donde cada imagen es una matriz 2-D de píxeles.
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(f"3-D: {arr}")
print(f"Dimensiones: {arr.ndim}")  # Número de dimensiones

# Creando una matriz multidimensional (de dimensiones superiores) con el parámetro ndmin
# El parámetro ndmin permite especificar el número mínimo de dimensiones que debe tener la matriz resultante.
# Esto es útil cuando se necesita trabajar con matrices de dimensiones específicas.
arr = np.array([1, 2, 3, 4], ndmin=5)
print(f"5-D: {arr}")
print(f"Dimensiones: {arr.ndim}")  # Número de dimensiones
