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

# Distribucion de Poisson
# La distribucion de Poisson es una distribucion discreta que describe el numero de eventos que ocurren en un intervalo de tiempo especifico. Por ejemplo, el numero de correos electronicos recibidos en un dia, el numero de llamadas recibidas en una linea telefonica, si alguien come dos veces al día, ¿cuál es la probabilidad de que coma tres veces? etc.
# Tiene 2 parametros: lam (tasa - numero promedio de eventos que ocurren en un intervalo de tiempo) y size (forma de la matriz).
print()
print("Distribucion de Poisson")

# Generando una distribicion de Poisson aleatoria de 1 x 10 para la ocurrencia 2.
x = random.poisson(lam=2, size=10)
print(
    f"Generando una distribicion de Poisson aleatoria de 1 x 10 para la ocurrencia 2 {x}")

# Visualizacion de la distribucion de Poisson
sns.histplot(random.poisson(lam=2, size=1000), kde=False)
plt.show()

# Distribucion uniforme
# Se utiliza para describir una distribucion donde todos los resultados son igualmente probables. Por ejemplo, lanzar un dado, numeros de loteria.
# Tiene 3 parametros: a (inicio), b (fin) y size (forma de la matriz).
print()
print("Distribucion uniforme")

# Generando una distribucion uniforme aleatoria de 2 x 3.
x = random.uniform(size=(2, 3))
print(f"Generando una distribucion uniforme aleatoria de 2 x 3 {x}")

# Visualizacion de la distribucion uniforme
sns.kdeplot(random.uniform(size=1000))
plt.show()

# Distribucion logistica
# Se utiliza para describir el crecimiento. Por ejemplo, la velocidad de crecimiento de una poblacion, la velocidad de crecimiento de una infeccion, redes neuronales.
# Tiene 3 parametros: loc (media), scale (desviacion estandar) y size (forma de la matriz).
print()
print("Distribucion logistica")

# Generando una distribucion logistica aleatoria de 2 x 3 con media 1 y desviacion estandar 2.
x = random.logistic(loc=1, scale=2, size=(2, 3))
print(
    f"Generando una distribucion logistica aleatoria de 2 x 3 con media 1 y desviacion estandar 2 {x}")

# Visualizacion de la distribucion logistica
sns.kdeplot(random.logistic(size=1000))
plt.show()

# Distribucion multinomial
# Se utiliza para describir escenarios donde se tienen multiples resultados posibles. Por ejemplo, lanzar un dado, donde el resultado puede ser 1, 2, 3, 4, 5, 6.
# Tiene 2 parametros: n (numero de intentos) y pvals (probabilidad de cada resultado).
print()
print("Distribucion multinomial")

# Generando una distribucion multinomial aleatoria de lanzamiento de dados.
x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(
    f"Generando una distribucion multinomial aleatoria de lanzamiento de dados {x}")

# Visualizacion de la distribucion multinomial
sns.histplot(random.multinomial(
    n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], size=1000), kde=False)
plt.show()

# Distribucion exponencial
# Se utiliza para describir el tiempo entre eventos en un proceso de Poisson, donde los eventos ocurren continuamente y de forma independiente a una tasa constante. Por ejemplo, el tiempo entre llegadas de autobuses, el tiempo entre llamadas telefonicas, el tiempo entre fallos de un sistema, etc.
# Tiene 2 parametros: scale (1/rate) y size (forma de la matriz).
print()
print("Distribucion exponencial")

x = random.exponential(scale=2, size=(2, 3))
print(
    f"Generando una distribucion exponencial aleatoria de 2 x 3 con media 1 y desviacion estandar 2 {x}")

# Visualizacion de la distribucion exponencial
sns.kdeplot(random.exponential(size=1000))
plt.show()

# Distribucion chi-cuadrado
# Se utiliza como una prueba de hipotesis para comparar la varianza de dos muestras. Tambien se utiliza en la regresion lineal. Por ejemplo, en la regresion lineal, se asume que los errores de la regresion siguen una distribucion normal, pero con la distribucion chi-cuadrado, podemos verificar si es correcto o no.
# Tiene 2 parametros: df (grados de libertad) y size (forma de la matriz).
print()
print("Distribucion chi-cuadrado")

x = random.chisquare(df=2, size=(2, 3))
print(
    f"Generando una distribucion chi-cuadrado aleatoria de 2 x 3 con media 1 y desviacion estandar 2 {x}")

# Visualizacion de la distribucion chi-cuadrado
sns.kdeplot(random.chisquare(df=2, size=1000))
plt.show()

# Distribucion de Rayleigh
# Se utiliza en comunicaciones inalambricas, para modelar la intensidad de una señal recibida, que puede ser debil o fuerte dependiendo de la distancia entre el transmisor y el receptor. Tambien se utiliza en la ingenieria de fiabilidad para modelar la vida util de un producto. Por ejemplo, la vida util de un televisor, la vida util de un telefono movil, etc.
# Tiene 2 parametros: scale (escala) y size (forma de la matriz).
print()
print("Distribucion de Rayleigh")

x = random.rayleigh(scale=2, size=(2, 3))
print(
    f"Generando una distribucion de Rayleigh aleatoria de 2 x 3 con media 1 y desviacion estandar 2 {x}")

# Visualizacion de la distribucion de Rayleigh
sns.kdeplot(random.rayleigh(size=1000))
plt.show()

# Distribucion de Pareto
# Se utiliza para modelar la distribucion de riqueza, ingresos, popularidad, etc. En general, se dice que el 80% de los ingresos pertenecen al 20% de la poblacion. Tambien se utiliza en la ingenieria de fiabilidad para modelar la vida util de un producto, donde se dice que el 80% de los fallos son causados por el 20% de los errores.
# Tiene 2 parametros: a (alfa) y size (forma de la matriz).
print()
print("Distribucion de Pareto")

x = random.pareto(a=2, size=(2, 3))
print(
    f"Generando una distribucion de Pareto aleatoria de 2 x 3 con media 1 y desviacion estandar 2 {x}")

# Visualizacion de la distribucion de Pareto
sns.histplot(random.pareto(a=2, size=1000), kde=False)
plt.show()

# Distribucion de Zipf
# Se utiliza para modelar la distribucion de palabras en un texto, popularidad de sitios web, frecuencia de palabras en un idioma, etc. En general, se dice que la palabra mas comun en un texto aparece dos veces mas que la segunda palabra mas comun, tres veces mas que la tercera palabra mas comun, etc. En general, la palabra n mas comun en un texto aparece 1/n veces que la palabra mas comun.
# Tiene 2 parametros: a (alfa) y size (forma de la matriz).
print()
print("Distribucion de Zipf")

x = random.zipf(a=2, size=(2, 3))
print(
    f"Generando una distribucion de Zipf aleatoria de 2 x 3 con media 1 y desviacion estandar 2 {x}")

# Visualizacion de la distribucion de Zipf
x = random.zipf(a=2, size=1000)
sns.histplot(x[x < 10], kde=False)
plt.show()
