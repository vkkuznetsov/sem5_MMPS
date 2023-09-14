import numpy as np

# Создаем матрицу 4x3
matrix = np.array([[3.72, 3.47, 3.06, 30.74],
                   [4.47, 4.10, 3.63, 36.80],
                   [4.96, 4.53, 4.01, 40.79]])

A = np.array([[3.72, 3.47, 3.06],
              [4.47, 4.10, 3.63],
              [4.96, 4.53, 4.01]])

b = np.array([30.74, 36.80, 40.79])

print("Исходная матрица:\n", matrix)


# def strait_go(A):
#     matrix = np.copy(A)
#     n = len(matrix)
#     for i in range(n):
#         for j in range(n, i - 1, -1):
#             matrix[i][j] /= matrix[i][i]
#         for k in range(i + 1, n):
#             for j in range(n, i - 1, -1):
#                 matrix[k][j] -= matrix[k][i] * matrix[i][j]
#     print("Полученная путем Гауссовскими преобразований матрица:\n", matrix)
#     return matrix


def gaussian_elimination_with_det(matrix):
    n = len(matrix)
    det = 1.0  # Инициализируем определитель единицей

    # Создаем копию исходной матрицы
    matrix_copy = np.copy(matrix)

    for i in range(n):
        for j in range(i + 1, n):
            factor = matrix_copy[j][i] / matrix_copy[i][i]
            for k in range(i, n + 1):  # до n + 1 потому что она не квадратная
                matrix_copy[j][k] -= factor * matrix_copy[i][k]

        # Обновляем определитель
        det *= matrix_copy[i][i]

        # Делаем единицу на главной диагонали
        matrix_copy[i] /= matrix_copy[i][i]

    return matrix_copy, det


upper_triangular_matrix, determinant = gaussian_elimination_with_det(matrix)
print("Верхнетреугольная матрица\n", upper_triangular_matrix)
print("Определитель матрицы =", determinant)


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


# 1 - 2

x = solution(upper_triangular_matrix)  # Получаем решение
print('Решение. Вектор x = \n', x)
print('-----------------------------------')
# Вычисление вектора невязки

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
    return augmented_matrix[:, n:]


A_inv = inverse_matrix(A)

print("Обратная матрица A^-1:\n", A_inv)
print('Перемноженные матрицы:\n', np.dot(A, A_inv))

# Шаг 3: Рассчитайте нормы матрицы A и A^-1
norm_A = np.linalg.norm(A, ord=2)  # Пример использования нормы 2 (L2-нормы)
norm_A_inv = np.linalg.norm(A_inv, ord=2)  # Пример использования нормы 2 (L2-нормы)

# Шаг 4: Рассчитайте число обусловленности ν
condition_number = norm_A * norm_A_inv
print("Число обусловленности ν =", condition_number)
