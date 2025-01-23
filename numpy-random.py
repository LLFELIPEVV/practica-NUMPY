from numpy import random
# Generando numeros aleatorios con numpy
x = random.randint(100)
print(f"Generando numeros aleatorios con numpy {x}")

# Generando un numero flotante aleatorio
x = random.rand()
print(f"Generando un numero flotante aleatorio {x}")

# Generando una matriz 1-D aleatoria
x = random.randint(100, size=(5))
print(f"Generando una matriz 1-D aleatoria {x}")

# Generando una matriz 2-D aleatoria
x = random.randint(100, size=(3, 5))
print(f"Generando una matriz 2-D aleatoria {x}")

# Generando una matriz 1-D aleatoria con numeros flotantes
x = random.rand(5)
print(f"Generando una matriz 1-D aleatoria con numeros flotantes {x}")

# Generando una matriz 2-D aleatoria con numeros flotantes
x = random.rand(3, 5)
print(f"Generando una matriz 2-D aleatoria con numeros flotantes {x}")

# Generando un numero aleatorio a partir de una matriz
x = random.choice([3, 5, 7, 9])
print(f"Generando un numero aleatorio a partir de una matriz {x}")

# Generando una nueva matriz aleatoria a partir de una matriz
x = random.choice([3, 5, 7, 9], size=(3, 5))
print(f"Generando una nueva matriz aleatoria a partir de una matriz {x}")

# Distribucion aleatoria de datos
# La distribucion de datos es una lista de todos los valores y la frecuencia de cada valor.
print()
print("Distribucion aleatoria de datos")

# Funcion de densidad de probabilidad es una funcion que describe la probabilidad de que una variable aleatoria tome un valor dado.
# Con choise() podemos especificar la probabilidad de cada valor.
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))
print(f"Generando una nueva matriz aleatoria a partir de una matriz {x}")

# Generando una matriz con una distribucion aleatoria 2-D
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))
print(f"Generando una matriz con una distribucion aleatoria 2-D {x}")
