import numpy as np

from math import log

# Ufuncs significa Funciones Universales que operan sobre el objeto ndarray.
# Se usan las ufuncs para implementar la vectorizacion en Numpy, lo que es mas rapido que iterar sobre elementos.
# Las ufuncs tambien aceptan argumentos adicionales como where(Matriz booleana o condicion), dtype(Para especificar el tipo de salida de la matriz) y out(Para almacenar el resultado en una matriz).
# La vectorizacion es la capacidad de organizar operaciones en matrices sin usar bucles, lo que hace que el codigo sea mas limpio y eficiente.
print("Ufuncs")

# Sumando dos matrices
# Sin ufunc
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []

for i, j in zip(x, y):
    z.append(i + j)
print(f"Suma de dos matrices sin ufunc: {z}")

# Con ufunc
z = np.add(x, y)
print(f"Suma de dos matrices con ufunc: {z}")

# Creando tu propia ufunc
# Se debe crear una funcion normal y luego usar el metodo frompyfunc() de la clase numpy para crear una ufunc.
# El metodo frompyfunc() toma los siguientes argumentos: function, inputs, outputs.
print()
print("Creando tu propia ufunc")


def myadd(x, y):
    return x + y


myadd = np.frompyfunc(myadd, 2, 1)
print(f"Suma de dos matrices con ufunc personalizada: {
      myadd([1, 2, 3, 4], [4, 5, 6, 7])}")

# Comprobando si una funcion es ufunc
print(f"Es ufunc: {type(np.add)}")
print(f"No es ufunc: {type(sum)}")

if type(np.add) == np.ufunc:
    print("Es ufunc")
else:
    print("No es ufunc")

# Aritmetica simple
# Se pueden realizar operaciones aritmeticas simples en matrices. Con la ayuda de funciones se puede realizar operaciones a cada elemento de la matriz.
# Todas las funciones se efectuan sobre arr1, por ejemplo arr1 + arr2, arr1 - arr2, arr1 * arr2, arr1 / arr2, arr1 % arr2, arr1 // arr2, arr1 ** arr2.
print()
print("Aritmetica simple")

# Suma
# La (add) funcion toma dos matrices como argumentos y devuelve la suma de ellas.
arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.add(arr1, arr2)

print(f"Suma de dos matrices: {newarr}")

# Resta
# La (subtract) funcion toma dos matrices como argumentos y devuelve la resta de ellas.
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.subtract(arr1, arr2)

print(f"Resta de dos matrices: {newarr}")

# Multiplicacion
# La (multiply) funcion toma dos matrices como argumentos y devuelve la multiplicacion de ellas.
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.multiply(arr1, arr2)

print(f"Multiplicacion de dos matrices: {newarr}")

# Division
# La (divide) funcion toma dos matrices como argumentos y devuelve la division de ellas.
# El resultado es en punto flotante.
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.divide(arr1, arr2)

print(f"Division de dos matrices: {newarr}")

# Potencia
# La (power) funcion toma dos matrices como argumentos y devuelve la potencia de ellas.
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 5, 6, 2, 4, 33])

newarr = np.power(arr1, arr2)

print(f"Potencia de dos matrices: {newarr}")

# Modulo
# La (mod) funcion toma dos matrices como argumentos y devuelve el resto de ellas.
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 4, 33])

newarr = np.mod(arr1, arr2)  # Tambien se puede usar np.remainder(arr1, arr2)

print(f"Resto de dos matrices: {newarr}")

# Cociente y modulo
# La (divmod) funcion toma dos matrices como argumentos y devuelve el cociente y el resto de ellas.
# Devuelve dos matrices la primera es el cociente y la segunda el resto.
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 4, 33])

newarr = np.divmod(arr1, arr2)

print(f"Cociente y resto de dos matrices: {newarr}")

# Valores absolutos
# La (absolute) funcion devuelve el valor absoluto de todos los elementos de la matriz.
# Se debe usar absolute() para no confundir abs() con funciones integradas de python math.abs().
arr = np.array([-1, -2, 1, 2, 3, -4])

newarr = np.absolute(arr)

print(f"Valor absoluto de una matriz: {newarr}")

# Redondeo de decimales
# Existen 5 formas de redondear decimales en Numpy.
print()
print("Redondeo de decimales")

# Truncamiento
# La (trunc) funcion redondea hacia 0.
# Se puede usar tanto trunc() como fix().
arr = np.trunc([-3.1666, 3.6667])

print(f"Truncamiento de una matriz: {arr}")

# Redondeo
# La (around) funcion redondea a la cantidad de decimales especificada. Redondea hacia arriba si el decimal es >= 5.
arr = np.around(3.1666, 2)

print(f"Redondeo de una matriz: {arr}")

# Piso
# La (floor) funcion redondea hacia abajo al entero mas cercano. Si el decimal es 3.1666, el resultado es 3.
arr = np.floor([-3.1666, 3.6667])

print(f"Piso de una matriz: {arr}")

# Techo
# La (ceil) funcion redondea hacia arriba al entero mas cercano. Si el decimal es 3.1666, el resultado es 4.
arr = np.ceil([-3.1666, 3.6667])

print(f"Techo de una matriz: {arr}")

# Numpy Logs
# Existen 4 funciones para calcular logs en Numpy.
print()
print("Numpy Logs")

# Logaritmo en base 2
# La (log2) funcion devuelve el logaritmo en base 2 de todos los elementos de la matriz.
arr = np.arange(1, 10)

print(f"Logaritmo en base 2 de una matriz: {np.log2(arr)}")

# Logaritmo en base 10
# La (log10) funcion devuelve el logaritmo en base 10 de todos los elementos de la matriz.
arr = np.arange(1, 10)

print(f"Logaritmo en base 10 de una matriz: {np.log10(arr)}")

# Logaritmo natural
# La (log) funcion devuelve el logaritmo natural de todos los elementos de la matriz.
arr = np.arange(1, 10)

print(f"Loragritmo natural de una matriz: {np.log(arr)}")

# Logaritmo en cualquier base
# Numpy no tiene una funcion especifica para calcular el logaritmo en cualquier base.
nplog = np.frompyfunc(log, 2, 1)

print(f"Logaritmo en cualquier base de una matriz: {nplog(100, 15)}")

# Sumas de Numpy
# La diferencia entre la suma y la adicion es que la adicion se hace entre 2 elementos, mientras que la suma se hace entre n elementos.
print()
print("Sumas de Numpy")

# Agregando los valores de arr1 a arr2
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

newarr = np.add(arr1, arr2)

print(f"Suma de dos matrices: {newarr}")  # [2 4 6]

# Suma los valores de arr1 y arr2
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

newarr = np.sum([arr1, arr2])

print(f"Suma de dos matrices: {newarr}")  # 12

# Suma sobre un eje
# Si se especifica el eje, se sumaran los elementos del eje especificado. Si se omite, se sumaran todos los elementos de la matriz. El eje 0 es para columnas y el eje 1 es para filas.
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

newarr = np.sum([arr1, arr2], axis=1)  # axis=1

print(f"Suma de dos matrices sobre un eje: {newarr}")  # [6 6]

newarr = np.sum([arr1, arr2], axis=0)  # axis=0

print(f"Suma de dos matrices sobre un eje: {newarr}")  # [2 4 6]

# Suma acumulativa
# La suma acumulativa es la suma de los elementos hasta el indice actual.
# La funcion (cumsum) devuelve la suma acumulativa de los elementos de la matriz.
arr = np.array([1, 2, 3])

newarr = np.cumsum(arr)

print(f"Suma acumulativa de una matriz: {newarr}")  # [1 3 6]

# Productos de Numpy
# La diferencia entre el producto y la multiplicacion es que la multiplicacion se hace entre 2 elementos, mientras que el producto se hace entre n elementos. El producto de 0 elementos es 1.
# La funcion (prod) devuelve el producto de los elementos de la matriz.
print()
print("Productos de Numpy")

# Multiplicando los valores de una matriz
arr = np.array([1, 2, 3, 4])

x = np.prod(arr)

print(f"Producto de una matriz: {x}")  # 24

# Encontrando el producto de los valores de dos matrices
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

x = np.prod([arr1, arr2])

print(f"Producto de dos matrices: {x}")  # 40320

# Producto sobre un eje
# Si se especifica el eje, se multiplicaran los elementos del eje especificado. Si se omite, se multiplicaran todos los elementos de la matriz. El eje 0 es para columnas y el eje 1 es para filas.
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

newarr = np.prod([arr1, arr2], axis=1)  # axis=1

print(f"Producto de dos matrices sobre un eje: {newarr}")  # [  24 1680]

newarr = np.prod([arr1, arr2], axis=0)  # axis=0

print(f"Producto de dos matrices sobre un eje: {newarr}")  # [ 5 12 21 32]

# Producto acumulativo
# El producto acumulativo es el producto de los elementos hasta el indice actual.
# La funcion (cumprod) devuelve el producto acumulativo de los elementos de la matriz.
arr = np.array([1, 2, 3, 4])

newarr = np.cumprod(arr)

print(f"Producto acumulativo de una matriz: {newarr}")  # [ 1  2  6 24]

# Diferencias de Numpy
# La diferencia entre la diferencia y la resta es que la resta se hace entre 2 elementos, mientras que la diferencia se hace entre n elementos.
# La funcion (diff) calcula la diferencia entre elementos sucesivos de la matriz.
print()
print("Diferencias de Numpy")

# Calculando la diferencia discreta de una matriz
arr = np.array([10, 15, 25, 5])  # x2 - x1, x3 - x2, x4 - x3

newarr = np.diff(arr)

print(f"Diferencia de una matriz: {newarr}")  # [ 5 10 -20]

# Calculando la diferencia discreta de una matriz n veces
arr = np.array([10, 15, 25, 5])

newarr = np.diff(arr, n=2)

print(f"Diferencia de una matriz n veces: {newarr}")  # [ 5  -30]

# Minimo comun multiplo (MCM) de Nympy
# El MCM es el numero mas pequeño que es divisible por dos numeros.
# La funcion (lcm) devuelve el MCM de los numeros especificados.
print()
print("Minimo comun multiplo (MCM) de Numpy")

# Encontrando el MCM de dos numeros.
num1 = 4
num2 = 6

x = np.lcm(num1, num2)

print(f"MCM de dos numeros: {x}")  # 12

# Encontrando el MCM en una matriz
# La funcion (lcm.reduce) devuelve el MCM de todos los elementos de la matriz.
arr = np.array([3, 6, 9])

x = np.lcm.reduce(arr)

print(f"MCM de una matriz: {x}")  # 18

# Encontrando el MCM de una matriz que contiene los elementos del 1 al 10
arr = np.arange(1, 11)
x = np.lcm.reduce(arr)

print(f"MCM de una matriz que contiene los elementos del 1 al 10: {x}")  # 2520

# Maximo comun divisor (MCD) de Numpy
# El MCD es el numero mas grande que es divisible por dos numeros.
# La funcion (gcd) devuelve el MCD de los numeros especificados.
print()
print("Maximo comun divisor (MCD) de Numpy")

# Encontrando el MCD de dos numeros.
num1 = 6
num2 = 9

x = np.gcd(num1, num2)

print(f"MCD de dos numeros: {x}")  # 3

# Encontrando el MCD en una matriz
# La funcion (gcd.reduce) devuelve el MCD de todos los elementos de la matriz.
arr = np.array([20, 8, 32, 36, 16])

x = np.gcd.reduce(arr)

print(f"MCD de una matriz: {x}")  # 4

# Funciones Trigonometricas de Numpy
# Numpy proporciona las ufuncs sin(), cos(), tan() que toman valores en radianes y producen los valores correspondientes.
print()
print("Funciones Trigonometricas de Numpy")

# Encontrando el valor de seno de PI/2
x = np.sin(np.pi/2)

print(f"Valor de seno de PI/2: {x}")  # 1.0 Radianes

# Encontrando el valor de seno de una matriz
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])

x = np.sin(arr)

# [1.         0.8660254  0.70710678 0.58778525]
print(f"Valor de seno de una matriz: {x}")

# Convirtiendo grados a radianes
# Los radianes = (pi/180) * (grados)
arr = np.array([90, 180, 270, 360])

x = np.deg2rad(arr)

# [1.57079633 3.14159265 4.71238898 6.28318531]
print(f"Convirtiendo grados a radianes: {x}")

# Convirtiendo radianes a grados
arr = np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])

x = np.rad2deg(arr)

print(f"Convirtiendo radianes a grados: {x}")  # [ 90. 180. 270. 360.]

# Encontrando angulos
# Numpy proporciona ufuncs arsin(), arcos(), arctan() que toman valores en radianes y producen los valores correspondientes.
x = np.arcsin(1.0)

print(f"Encontrando angulos: {x}")  # 1.5707963267948966

# Angulos para todos los valores de seno de una matriz
arr = np.array([1, -1, 0.1])
x = np.arcsin(arr)

# [ 1.57079633 -1.57079633  0.10016742]
print(f"Angulos para todos los valores de seno de una matriz: {x}")

# Encontrando hipotenusas
# Numpy proporciona la ufunc hypot() que toma los valores de la base y de la perpendicular y produce las hipotenusas.
base = 3
perp = 4

x = np.hypot(base, perp)

print(f"Encontrando hipotenusas: {x}")  # 5.0

# Funciones Hiperbolicas de Numpy
# Numpy proporciona las ufuncs sinh(), cosh(), tanh() que toman valores en radianes y producen los valores correspondientes.
print()
print("Funciones Hiperbolicas de Numpy")

# Encontrando el valor de seno hiperbolico de PI/2
x = np.sinh(np.pi/2)

print(f"Valor de seno hiperbolico de PI/2: {x}")  # 1.0 Radianes

# Encontrando los valores de coseno en una matriz
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])

x = np.cosh(arr)

# [2.50917848 1.60028686 1.32460909 1.20397209]
print(f"Valor de coseno en una matriz: {x}")

# Encontrando angulos hiperbolicos
# Numpy proporciona ufuncs arcsinh(), arccosh(), arctanh() que producen los valores correspondientes en radianes.
x = np.arcsinh(1.0)

print(f"Encontrando angulos hiperbolicos: {x}")  # 0.881373587019543

# Angulos para todos los valores de tanh de una matriz
arr = np.array([0.1, 0.2, 0.5])

x = np.arctanh(arr)

# [0.10033535 0.20273255 0.54930614]
print(f"Angulos para todos los valores de tanh de una matriz: {x}")

# Operaciones de conjuntos de Numpy
print()
print("Operaciones de conjuntos de Numpy")

# Creando un conjunto
arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])

x = np.unique(arr)

print(f"Creando un conjunto: {x}")  # [1 2 3 4 5 6 7]

# Encontrando la unión de dos conjuntos
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

newarr = np.union1d(arr1, arr2)

print(f"Encontrando la unión de dos conjuntos: {newarr}")  # [1 2 3 4 5 6]

# Encontrando la intersección de dos conjuntos
# El argumento assume_unique acelera el calculo, siempre debe estar activado cuando se trabaja con conjuntos.
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

newarr = np.intersect1d(arr1, arr2, assume_unique=True)

print(f"Encontrando la intersección de dos conjuntos: {newarr}")  # [3 4]

# Encontrando la diferencia entre dos conjuntos
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])

newarr = np.setdiff1d(set1, set2, assume_unique=True)

print(f"Encontrando la diferencia entre dos conjuntos: {newarr}")  # [1 2]

# Encontrando la diferencia simetrica entre dos conjuntos
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])

newarr = np.setxor1d(set1, set2, assume_unique=True)

print(f"Encontrando la diferencia simetrica entre dos conjuntos: {
      newarr}")  # [1 2 5 6]
