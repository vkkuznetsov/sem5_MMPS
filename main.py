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
print('----------------------------------------')


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

# 1 - 3, 5
def gaussian_elimination_with_det(matrix):
    n = len(matrix)
    det = 1.0  # Инициализируем определитель единицей

    # Создаем копию исходной матрицы
    matrix_copy = np.copy(matrix)

    for i in range(n):
        for j in range(i + 1, n):
            factor = matrix_copy[j][i] / matrix_copy[i][i]
            for k in range(i, len(matrix[i])):  # до n + 1 потому что она не квадратная
                matrix_copy[j][k] -= factor * matrix_copy[i][k]

        # Обновляем определитель
        det *= matrix_copy[i][i]

        # Делаем единицу на главной диагонали
        matrix_copy[i] /= matrix_copy[i][i]

    return matrix_copy, det


upper_triangular_matrix, determinant = gaussian_elimination_with_det(matrix)
print("Верхнетреугольная матрица\n", upper_triangular_matrix)
print("Определитель матрицы =", determinant)

det_true = np.linalg.det(A)
print('Определитель который получится у Numpy', det_true)


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


x = solution(upper_triangular_matrix)  # Получаем решение
print('Решение. Вектор x = \n', x)
print('----------------------------------------')


# 4
# Вычисление вектора невязки

# Вычисление вектора невязки алгоритмом
def vector_unchain(matrix, x):
    r = np.zeros(len(matrix))
    for col in range(len(matrix)):
        r[col] = b[col]
        for row in range(len(matrix)):
            r[col] -= matrix[col][row] * x[row]
    return r

# Вычисление нормы inf\
r = vector_unchain(A, x)
norm_inf = max(abs(r))
print(r)
print('Норма inf', norm_inf)
print('----------------------------------------')


# 6
# Нахождение обратной матрицы методом Гаусса
def inverse_matrix(a):
    n = a.shape[0]

    # Создаем расширенную матрицу [A | E]
    augmented_matrix = np.hstack((a, np.identity(n)))

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
print('----------------------------------------')

# Рассчитать нормы матрицы A и A^-1


def l2_norm_matrix(matrix):
    sum_of_squares = 0.0

    for row in matrix:
        for element in row:
            sum_of_squares += element ** 2

    l2_norm = np.sqrt(sum_of_squares)
    return l2_norm


norm_A_L2 = l2_norm_matrix(A)
norm_A_inv_L2 = l2_norm_matrix(A_inv)
condition_number_L2 = norm_A_L2 * norm_A_inv_L2
print( 'Число обусловленности по л2:',condition_number_L2)

# Сравнение с результатами numpy
A_inv_np = np.linalg.inv(A)
condition_number_np = np.linalg.cond(A)
print("Число обусловленности (numpy):", condition_number_np)
