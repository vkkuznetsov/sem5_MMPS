import numpy as np

# Создаем матрицу 4x3
matrix = np.array([[3.72, 3.47, 3.06, 30.74],
                   [4.47, 4.10, 3.63, 36.80],
                   [4.96, 4.53, 4.01, 40.79]])
print("Исходная матрица\n", matrix)


def strait_go(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n, i - 1, -1):
            matrix[i][j] /= matrix[i][i]
        for k in range(i + 1, n):
            for j in range(n, i - 1, -1):
                matrix[k][j] -= matrix[k][i] * matrix[i][j]
    print("Полученная путем Гауссовскими преобразований матрица:\n", matrix)


def solution(matrix):
    n = len(matrix)
    # Создаем список для хранения решения
    x = [0] * n

    # Метод обратной подстановки
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i][n]  # Значение правой части
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]  # Вычитаем уже найденные переменные
    return x


strait_go(matrix)
x = solution(matrix)  # Получаем решение

# Создание вектора правой части b
b = matrix[:, -1]

# Вычисление вектора невязки
print(matrix[:, :-1])
r = b - np.dot(matrix[:, :-1], x)

# Вычисление норм вектора невязки
for i in range(len(matrix)):
    r[i] = b[i]
    for j in range(len(matrix)):
        r[i] -= matrix[i][j] * x[j]
print(r)
norm_1 = np.linalg.norm(r, ord=1)  # Норма 1
norm_inf = np.linalg.norm(r, ord=np.inf)  # Норма ∞
norm_2 = np.linalg.norm(r, ord=2)  # Норма 2

print("Решение x:\n", x)
print("Вектор невязки r:\n", r)
print("Норма 1:", norm_1)
print("Норма ∞:", norm_inf)
print("Норма 2:", norm_2)

# Вычисление определителя
A = np.array([[3.72, 3.47, 3.06],
              [4.47, 4.10, 3.63],
              [4.96, 4.53, 4.01]])


# Нахождение определителя методом Гаусса
def gaussian_elimination(A):
    matrix = np.copy(A)
    n = len(matrix)

    det = 1
    for i in range(n):
        if matrix[i][i] == 0:
            return 0
        det *= matrix[i][i]
        for j in range(i + 1, n):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, n):
                matrix[j][k] -= factor * matrix[i][k]
    return det


det = gaussian_elimination(A)
print('Определитель который получился у меня', det)
det_true = np.linalg.det(A)
print('Определитель который получится у Numpy', det_true)


# Нахождение обратной матрицы методом Гаусса
def inverse_matrix(A):
    n = A.shape[0]

    # Создаем расширенную матрицу [A | E]
    augmented_matrix = np.hstack((A, np.identity(n)))

    for col in range(n):
        # Находим максимальный элемент в текущем столбце
        max_val = augmented_matrix[col, col]
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented_matrix[row, col]) > abs(max_val):
                max_val = augmented_matrix[row, col]
                max_row = row

        # Обменять строки
        augmented_matrix[[col, max_row]] = augmented_matrix[[max_row, col]]

        # Привести главную диагональ к 1
        augmented_matrix[col] /= augmented_matrix[col, col]

        # Обнулить элементы под главной диагональю
        for row in range(n):
            if row != col:
                factor = augmented_matrix[row, col]
                augmented_matrix[row] -= factor * augmented_matrix[col]
    print(augmented_matrix)
    return augmented_matrix[:, n:]


# Пример использования
A = np.array([[3.72, 3.47, 3.06],
              [4.47, 4.10, 3.63],
              [4.96, 4.53, 4.01]])

A_inv = inverse_matrix(A)

print("Обратная матрица A^-1:")
print(A_inv)

print(A)
a_inv = np.linalg.inv(A)
print('rev matrix', a_inv)

result = np.dot(A, a_inv)
print(result)
