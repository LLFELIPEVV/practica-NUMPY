import numpy as np

# Creando un array en Numpy
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# Imprimiendo la versión de Numpy
print(f"Versión: {np.__version__}")

# Verificando el tipo del array
print(f"Tipo: {type(arr)}")

# Creando un array a partir de una tupla
arr = np.array((1, 2, 3, 4, 5))
print(arr)

# Matrices 0-D: también conocidas como escalares, son matrices de cero dimensiones.
arr = np.array(42)
print(f"0-D: {arr}")
print(f"Dimensiones: {arr.ndim}")

# Matrices 1-D: también conocidas como vectores, son matrices de una sola dimensión.
arr = np.array([1, 2, 3, 4, 5])
print(f"1-D: {arr}")
print(f"Dimensiones: {arr.ndim}")

# Matrices 2-D: también conocidas como matrices bidimensionales, son arreglos que contienen matrices 1-D como sus elementos.
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2-D: {arr}")
print(f"Dimensiones: {arr.ndim}")

# Matrices 3-D: también conocidas como arreglos tridimensionales, son matrices que contienen matrices 2-D como sus elementos.
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(f"3-D: {arr}")
print(f"Dimensiones: {arr.ndim}")

# Creando una matriz multidimensional (de dimensiones superiores) con el parámetro ndmin
arr = np.array([1, 2, 3, 4], ndmin=5)
print(f"5-D: {arr}")
print(f"Dimensiones: {arr.ndim}")

# Accediendo a elementos de un array de NumPy
print(f"Primer elemento: {arr[0, 0, 0, 0, 0]}")
print(f"Segundo elemento: {arr[0, 0, 0, 0, 1]}")
print(f"Suma de los elementos: {arr[0, 0, 0, 0, 0] + arr[0, 0, 0, 0, 1]}")

# Slice de un ndarray
print(f"Slice: {arr[0, 0, 0, 0, 0:-1:2]}")
print(f"Slice: {arr[0, 0, 0, 0, 0::2]}")

# Tipos de datos en NumPy
"""
i - entero
b - booleano
u - entero sin signo
f - flotante
c - flotante complejo
m - timedelta
M - datetime
O - objeto
S - cadena
U - cadena Unicode
V - otro tipo (vacío)
"""

# Comprobación de tipo de dato de una matriz
print(f"Tipo de dato: {arr.dtype}")

# Creando una matriz con un tipo de dato específico
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
print(f"Tipo de dato: {arr.dtype}")

# Definiendo el tamaño del dato
arr = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(f"Tipo de dato: {arr.dtype}")

# Cambiando el tipo de dato de una matriz
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print(newarr)
print(f"Tipo de dato: {newarr.dtype}")
