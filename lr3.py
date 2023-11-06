import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# Определение устройства для вычислений (GPU, если доступно, иначе CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Проверка доступности CUDA (не используется)
torch.cuda.is_available()

# Определение функции активации
activation_function = nn.Tanh()

# Определение функции, которую будем аппроксимировать
def myfun(x):
    # return 3 * np.sin((x - 2) * 2) - 2
    return np.sin(x/2)

# Функция для аппроксимации 1D функции
def approx_1d_function(x_train, x_eval, units, epochs):
    # Генерация меток для тренировочных данных
    y_train = myfun(x_train)
    # Масштабирование данных для нормализации
    x_scaler = MinMaxScaler(feature_range=((-1, 1)))
    y_scaler = MinMaxScaler(feature_range=((-1, 1)))

    x_scaled = x_scaler.fit_transform(x_train)
    y_scaled = y_scaler.fit_transform(y_train)
    x_eval_scaled = x_scaler.transform(x_eval)

    # Построение и применение MLP
    _, result_eval = train_model_simple(x_scaled, y_scaled, x_eval_scaled, units, epochs)

    # Возвращение данных к исходному диапазону
    res_rescaled = y_scaler.inverse_transform(result_eval)

    # Расчет меток для оценочных данных
    y_eval = myfun(x_eval)
    return x_eval, res_rescaled, y_eval

# Функция для обучения модели
def train_model_simple(x_train, y_train, x_eval, units, epochs):
    # Преобразование данных в тензоры PyTorch и перемещение на выбранное устройство
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    x_eval_tensor = torch.from_numpy(x_eval).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)

    # Определение линейных слоев с заданным количеством нейронов
    layer1 = nn.Linear(x_train.shape[1], units).to(device)
    layer2 = nn.Linear(units, 1).to(device)

    # Сбор параметров обоих слоев для оптимизации
    parameters = list(layer1.parameters()) + list(layer2.parameters())

    # Определение оптимизатора и функции потерь
    optimizer = optim.Adam(parameters)
    loss_fn = nn.MSELoss(reduction='mean')

    # Основной цикл обучения
    for epoch in range(epochs):
        # Прямой проход
        yhat = layer2(activation_function(layer1(x_train_tensor)))

        # Расчет потерь
        loss = loss_fn(yhat, y_train_tensor)

        # Обратный проход (расчет градиентов)
        loss.backward()

        # Шаг оптимизации
        optimizer.step()

        # Сброс градиентов
        optimizer.zero_grad()

    # Применение обученной модели к оценочным данным
    yhat_eval = layer2(activation_function(layer1(x_eval_tensor)))

    return yhat.detach().cpu().numpy(), yhat_eval.detach().cpu().numpy()

# Функция для визуализации результатов
def plot_1d_function(x_train, x_eval, predictions, labels, units, epochs):
    # Создание фигуры и осей
    fig = plt.figure(1, figsize=(18,6))
    ax = fig.add_subplot(1, 2, 1)
    ax.axvspan(x_train.flatten()[0], x_train.flatten()[-1], alpha=0.15, color='limegreen')
    # Построение графиков
    plt.plot(x_eval, myfun(x_eval), '-', color='royalblue', linewidth=1.0)
    plt.plot(x_eval, predictions, '-', label='output', color='darkorange', linewidth=2.0)
    plt.plot(x_train, myfun(x_train), '.', color='royalblue')
    plt.grid(which='both')
    plt.rcParams.update({'font.size':14})
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('%d neurons in hidden layer with %d epochs of training' % (units, epochs))
    plt.legend(['Function f(x)', 'MLP output g(x)', 'Training set'])
    ax = fig.add_subplot(1, 2, 2)
    ax.axvspan(x_train.flatten()[0], x_train.flatten()[-1], alpha=0.15, color='limegreen')
    # Построение графика ошибки
    plt.plot(x_eval, np.abs(predictions-myfun(x_eval)), '-', label='output', color='firebrick', linewidth=2.0)
    plt.grid(which='both')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Absolute difference between prefiction and actual function')
    plt.legend(['Error |f(x)-g(x)|'])
    plt.show()

# Установка параметров
# batch_size_train = 128
batch_size_train = 64
batch_size_eval = 128

# Генерация тренировочных и оценочных данных
x_train_fix = np.linspace(0, 10, num=batch_size_train).reshape(-1,1)
x_eval_fix = np.linspace(0, 10, num=batch_size_eval).reshape(-1,1)

# Установка параметров модели
# units = 15
units = 5
epochs = 10000

# Выполнение аппроксимации и визуализация результатов
x, predictions, labels = approx_1d_function(x_train=x_train_fix, x_eval=x_eval_fix, units=units, epochs=epochs)
plot_1d_function(x_train_fix, x_eval_fix, predictions, labels, units, epochs)