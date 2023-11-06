import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial.distance import euclidean

np.random.seed(0)

# Генерируем тренировочные данные
N = 60 # Общее количество точек
n_red = N // 3  # Количество красных точек
n_blue = N // 3  # Количество синих точек
n_green = N - n_red - n_blue  # Количество зеленых точек

red_points = np.random.rand(n_red, 2) * 5 - np.array([5, 5]) # Красные точки в левом нижнем углу
blue_points = np.random.rand(n_blue, 2) * 10 - np.array([2.5, 2.5])  # Синие точки в центре
green_points = np.random.rand(n_green, 2) * 5 + np.array([5, 5])  # Зеленые точки в правом верхнем углу

# Объединяем точки в один массив
X_train = np.vstack((red_points, blue_points, green_points))
y_train = np.array(['red'] * n_red + ['blue'] * n_blue + ['green'] * n_green)

# Визуализируем тренировочные данные
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Тренировочные данные')
plt.show()

# Создаем тестовую выборку
n_test = int(0.1 * N)  # 10% от общего количества точек 0.2
X_test = np.random.rand(n_test, 2) * 10

# Создаем соответствующие метки для тестовых данных
y_test = np.random.choice(['red', 'blue', 'green'], size=n_test)

# Инициализируем классификатор k-NN
k = 3  # Количество соседей
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Визуализируем результаты
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Тренировочные и тестовые данные')

# Выводим окружности для k ближайших соседей
for i in range(n_test):
    distances, indices = knn.kneighbors([X_test[i]], n_neighbors=k)
    circle_radius = max([euclidean(X_train[j], X_test[i]) for j in indices[0]])
    circle = plt.Circle(X_test[i], circle_radius, color='gray', fill=False)
    plt.gca().add_patch(circle)

plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Тестовые данные с окружностями k ближайших соседей')
plt.show()

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Вывод матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
