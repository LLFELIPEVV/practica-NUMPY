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

# Creando una copia de una matriz
# La copia de la matriz no afecta a la original ni la original afecta a la copia
x = arr.copy()
x[0] = 42
print(f"Original: {arr}")
print(f"Copia: {x}")

# Creando una vista de una matriz
# La vista de la matriz afecta a la original y la original afecta a la copia
x = arr.view()
arr[0] = 42
print(f"Original: {arr}")
print(f"Vista: {x}")

# Comprobando la autoria de los datos
x = arr.copy()
y = arr.view()

print(f"Copia: {x.base}")  # None
print(f"Vista: {y.base}")  # [42  2.1  3.1]

# Obteniendo la forma de una matriz
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Forma: {arr.shape}")

# Remodelando una matriz
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Forma (12)
print(arr.shape)

newarr = arr.reshape(4, 3)  # Forma (4, 3)
print(newarr)
print(newarr.shape)

newarr = arr.reshape(2, 3, 2)  # Forma (2, 3, 2)
# Pasar -1 como argumento a una dimensión significa que se calcula automáticamente
# Pasar unicamente -1 como argumento aplanara la matriz en una sola dimensión
print(newarr)
print(newarr.shape)

# Iteraciones de matricez
# Iterando una matriz 1-D
# Salto de línea
print()
print("Iteracion de matrices")
arr = np.array([1, 2, 3])

for x in arr:
    print(x)

# Iterando una matriz 2-D
arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
    for y in x:
        print(y)

# Iterando una matriz 3-D
# Es mejor realizar una funcion que disminuya las damensiones de la matriz y vaya entregando los valores
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print("3-D")

for x in arr:
    for y in x:
        for z in y:
            print(z)

# Iterando una matriz con nditer()
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
    print(x)

# Iterando una matriz con buffer
arr = np.array([1, 2, 3])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
    print(x)

# Iterando una matriz saltando elementos
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
    print(x)

# Iterando una matriz con enumeración de índices
arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr):
    print(idx, x)

# Unir matrices en Numpy
print()
print("Unir matrices en Numpy")

# Uniendo matrices en el eje 0
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Poner axis=0 o no poner nada es lo mismo
arr = np.concatenate((arr1, arr2), axis=0)
print("Unir matrices en el eje 0")
print(arr)

# Uniendo matrices en el eje 1
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)
print("Unir matrices en el eje 1")
print(arr)

# Uniendo matrices mediante funciones de pila (stack)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)
print("Unir matrices mediante funciones de pila (stack)")
print(arr)

# Apilando matrices verticalmente
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

arr = np.vstack((arr1, arr2))
print("Apilando matrices en vertical")
print(arr)

# Apilando matrices horizontalmente
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

arr = np.hstack((arr1, arr2))
print("Apilando matrices en horizontal")
print(arr)

# Apilando matrices en profundidad
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.dstack((arr1, arr2))
print("Apilando matrices en profundidad")
print(arr)
print(arr.shape)

# Dividiendo matrices en Numpy
# Se usa la funcion array_split() para dividir una matriz en submatrices
# array_split() divide la matriz sin importar el numero de elementos
# Tambien esta el metodo split() pero no ajusta el numero de elementos
print()
print("Dividiendo matrices en Numpy")

arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(f"Dividiendo una matriz en 3 submatrices: {newarr}")
print(f"Primer submatriz: {newarr[0]}")
print(f"Segunda submatriz: {newarr[1]}")
print(f"Tercera submatriz: {newarr[2]}")

# Tambien se les puede especificar el eje donde se dividira la matriz
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3, axis=1)
print(f"Dividiendo una matriz en 3 submatrices en el eje 1: {newarr}")

# El hsplit() es equivalente a split() con axis=1, el vsplit() es equivalente a split() con axis=0
newarr = np.hsplit(arr, 2)
print(f"Dividiendo una matriz en 2 submatrices en el eje 1: {newarr}")

# Buscando elementos en una matriz
# Para buscar en una matriz se usa el metodo where()
print()
print("Buscando elementos en una matriz")

arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(f"Indices de elementos que son 4: {x}")

x = np.where(arr % 2 == 0)
print(f"Indices de elementos que son pares: {x}")

x = np.where(arr % 2 == 1)
print(f"Indices de elementos que son impares: {x}")

# Se puede usar el metodo searchsorted() para hacer una busqueda binaria en una matriz
# Por defecto inicia de izquierda a derecha pero se puede modificar el parametro side='right' para iniciar de derecha a izquierda
x = np.searchsorted(arr, 1)
print(
    f"Indice donde se deberia insertar el valor 5 para mantener el orden: {x}")

x = np.searchsorted(arr, 1, side='right')
print(
    f"Indice donde se deberia insertar el valor 5 para mantener el orden: {x}")

# Tambien se puede valores multiples
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, [6, 7, 9])
print(
    f"Indices donde se deberian insertar los valores 6, 7 y 9 para mantener el orden: {x}")

# Ordenando matrices en Numpy
# Se usa el metodo sort() para ordenar una matriz, el cual usa un algoritmo de ordenamiento de tipo quicksort
# Tambien acepta otro tipo de algoritmos de ordenamiento como 'mergesort' y 'heapsort'
print()
print("Ordenando matrices en Numpy")

arr = np.array([3, 2, 0, 1])
print(f"Matriz numerica ordenada: {np.sort(arr)}")

arr = np.array(['banana', 'cherry', 'apple'])
print(f"Matriz de cadenas ordenada: {np.sort(arr)}")

# Ordenando una matriz de booleanos
arr = np.array([True, False, True])
print(f"Matriz de booleanos ordenada: {np.sort(arr)}")

# Ordenando una matriz 2-D
arr = np.array([[3, 2, 4], [5, 0, 1]])
print(f"Matriz 2-D ordenada: {np.sort(arr)}")

# Filtrando matrices en Numpy
# Si el valor de un indice es True se incluye en la matriz resultante, si es False no se incluye
print()
print("Filtrando matrices en Numpy")

arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]

print(f"Matriz filtrada: {newarr}")

# Filtrando matrices
arr = np.array([41, 42, 43, 44])
filter_arr = []

for element in arr:
    if element > 42:
        filter_arr.append(True)
    else:
        filter_arr.append(False)

newarr = arr[filter_arr]
print(f"Matriz filtrada: {newarr}")

# Creando una matriz de pares
arr = np.array([41, 42, 43, 44])
filter_arr = []

for element in arr:
    if element % 2 == 0:
        filter_arr.append(True)
    else:
        filter_arr.append(False)

newarr = arr[filter_arr]
print(f"Matriz de pares: {newarr}")

# Creando filtros directamente desde una matriz
arr = np.array([41, 42, 43, 44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(f"Matriz filtrada: {newarr}")

# Creando filtro de pares directamente desde una matriz
arr = np.array([41, 42, 43, 44])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]
print(f"Matriz de pares: {newarr}")
