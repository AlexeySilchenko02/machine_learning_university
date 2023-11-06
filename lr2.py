# pip install numpy matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''# Генерация случайных данных
np.random.seed(0)  # Для воспроизводимости результатов
m = 300  # Количество точек
k = 3    # Количество кластеров

# Генерируем случайные точки в пространстве R[0, 10]xR[0, 10]
X = np.random.rand(m, 2) * 10

# Визуализация сгенерированных данных
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Сгенерированные данные')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()'''

# Генерация случайных данных в углах квадрата
np.random.seed(0)
m = 90  # Количество точек
k = 3    # Количество кластеров

# Генерируем данные в углах квадрата
X = np.zeros((m, 2))
X[:30, :] = np.random.rand(30, 2) * 2  # Левый верхний угол
X[30:60, :] = np.random.rand(30, 2) * 2 + [3, 0]  # Правый верхний угол
X[60:, :] = np.random.rand(30, 2) * 2 + [0, 3]  # Левый нижний угол

# Визуализация сгенерированных данных
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Сгенерированные данные')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Применяем алгоритм k-средних
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Визуализация результатов кластеризации
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r', label='Центроиды')
plt.title('Результаты кластеризации методом k-средних')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
