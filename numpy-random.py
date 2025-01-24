import seaborn as sns
import matplotlib.pyplot as plt

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

# Permutaciones aleatorias
# El metodo shuffle() se utiliza para mezclar una matriz y el metodo permutation() se utiliza para devolver una nueva matriz mezclada.
print()
print("Permutaciones aleatorias")

# Mezcla de matrices
arr = [1, 2, 3, 4, 5]
random.shuffle(arr)
print(f"Permutaciones aleatorias {arr}")

# Permutacion de matriz
arr = [1, 2, 3, 4, 5]
print(f"Permutacion de matriz {random.permutation(arr)}")

# Distribucion normal (Gaussina)
# Se usa el metodo normal() para obtener una distribucion normal (gaussiana) de los datos.
# Se ajusta a la distribucion de probabilidad de muchos eventos, como la altura de las personas, la velocidad de los automoviles, puntuaciones de CI, ritmo cardiaco, etc.
# Tiene 3 parametros: loc (media), scale (desviacion estandar) y size (forma de la matriz).
print()
print("Distribucion normal (Gaussina)")

# Generando una distribucion normal aleatoria de tamaño 2x3
x = random.normal(size=(2, 3))
print(f"Generando una distribucion normal aleatoria de tamaño 2x3 {x}")

# Generando una distribucion normal aleatoria de tamaño 2x3 con media 1 y desviacion estandar 2
x = random.normal(loc=1, scale=2, size=(2, 3))
print(
    f"Generando una distribucion normal aleatoria de tamaño 2x3 con media 1 y desviacion estandar 2 {x}")

# Visualizacion de la distribucion normal
sns.kdeplot(random.normal(size=1000))
plt.show()

# Distribucion binomial
# Describe escenarios con solo 2 posibles resultados, como lanzar una moneda, que puede ser cara o cruz.
# Tiene 3 parametros: n (numero de intentos), p (probabilidad de cada intento) y size (forma de la matriz).
print()
print("Distribucion binomial")

# Generando valores de 10 lanzamientos de moneda, con 0.5 probabilidad de que salga cara
x = random.binomial(n=10, p=0.5, size=10)
print(
    f"Generando valores de 10 lanzamientos de moneda, con 0.5 probabilidad de que salga cara {x}")

# Visualizacion de la distribucion binomial
sns.histplot(random.binomial(n=10, p=0.5, size=1000), kde=False)
plt.show()

# Diferencia entre distribucion normal y binomial
# La principal diferencia es que la distribucion normal es continua mientras que la distribucion binomial es discreta. Pero si hay suficientes intentos, la distribucion binomial se asemejara a la distribucion normal.

sns.kdeplot(random.normal(loc=50, scale=5, size=1000), label='normal')
sns.kdeplot(random.binomial(n=100, p=0.5, size=1000), label='binomial')
plt.legend()
plt.show()
