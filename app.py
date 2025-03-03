# 1. Число обсусловленности 3х
# 2.а. Критерии сходимости
# 2.б Расчёт до заданной точности
# 3. Оценка для заданное epsilon.

# Ax = b
# Постановка задачи в рамках разработки:
# На вход: матрица A, вектор b, точность epsilon
# Выход: вектор x, график с показанием каждой итерации, условие подходимости критерия.

from methods.easy_iterations import EasyIterations
from methods.jacobi_method import JacobiMethod
from methods.gauss_zeidel import GaussZeidelMethod

import numpy as np

A_matrix = np.array([
    [8, 4, 0, 1],
    [0, 6, 1, 4],
    [6, 1, 9, 1],
    [6, 1, 5, 10]
])
b_vector = np.array([14, 11, 3, 10])

epsilon = 10 ** (-3)




A_matrix = np.array([
    [14, 6, 3, 5],
    [9, 12, 2, 0],
    [5, 1, 13, 5],
    [4, 5, 5, 15]
])
b_vector = np.array([12, 5, 2, 10])

print()
print("\nEasy iteration method:")
# method = EasyIterations(A_matrix, b_vector, 10 ** (-3), False)
# method = JacobiMethod(A_matrix, b_vector, epsilon, False)
method = GaussZeidelMethod(A_matrix, b_vector, epsilon, True)
print("[INFO] Is solvable definitely:", method.base_check_conditions())
print()
print("[INFO] Condition numbers: ", method.count_condition_numbers())
print()
print("[INFO] Is applicable:", method.check_condition_of_usage())
print()
print("[INFO] Numerical solution: ", method.numerical_solution())


from pprint import pp
pp(method.past_solutions.get_dictionary())
pp(method.past_solutions.show_plot())

print([3931/4336, 3357/2168, -547/1084, 150/271])
