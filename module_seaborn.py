import matplotlib.pyplot as plt
import seaborn as sns

# Visualizar distribuciones con seaborn
# Trazando un diagrama de distribucion
sns.histplot([0, 1, 2, 3, 4, 5], kde=True)
plt.show()

# Trazando un diagrama de distribucion sin el histograma
sns.kdeplot([0, 1, 2, 3, 4, 5])
plt.show()
