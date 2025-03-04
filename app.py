# 1. Число обсусловленности 3х
# 2.а. Критерии сходимости
# 2.б Расчёт до заданной точности
# 3. Оценка для заданное epsilon.

# Ax = b
# Постановка задачи в рамках разработки:

# На вход: матрица A, вектор b, точность epsilon, количество случайных векторов.
# Выход: 
# 0. Условие подходимости критерия.
# 1. Вектор x, 
# 2. График с показанием зависимости количества итераций от нормы разности 
#    точного решения и начального решения.

from methods.easy_iterations import EasyIterations
from methods.jacobi_method import JacobiMethod
from methods.gauss_zeidel import GaussZeidelMethod
from methods_interface import AnyNumericalMethod

import numpy as np

A_matrix = np.array([
    [8, 4, 0, 1],
    [0, 6, 1, 4],
    [6, 1, 9, 1],
    [6, 1, 5, 10]
])
# Вектор свободных членов
b_vector = np.array([14, 11, 3, 10])

# Точное решение
precise_answer = np.array(
    [
        3931 / 4336, 
        3357 / 2168, 
        -547 / 1084, 
        150 / 271
    ]
)


methods = {
    "EasyIterations": [
        EasyIterations(A_matrix, b_vector, 10**(-1), True),
        EasyIterations(A_matrix, b_vector, 10**(-2), False),
        EasyIterations(A_matrix, b_vector, 10**(-3), False),
    ],
    "Jacobi": [
        JacobiMethod(A_matrix, b_vector, 10**(-1), False),
        JacobiMethod(A_matrix, b_vector, 10**(-2), False),
        JacobiMethod(A_matrix, b_vector, 10**(-3), False),
    ],
    "GaussZeidel": [
        GaussZeidelMethod(A_matrix, b_vector, 10**(-1), False),
        GaussZeidelMethod(A_matrix, b_vector, 10**(-2), False),
        GaussZeidelMethod(A_matrix, b_vector, 10**(-3), True),
    ]
}


for method_name, methods_realizations in methods.items():
    method: AnyNumericalMethod
    for method in methods_realizations:
        print("[START]")
        is_ok = method.base_check_conditions()
        if not is_ok:
            print("IT IS NOT OK BACICALLY, SKIP")
            continue
        is_ok = method.check_condition_of_usage()
        if not is_ok:
            print("IT IS NOT OK BACICALLY, SKIP")
            continue

        print(f"CHECK GRAPHICS. METHOD __{method_name}__; PRESISION = {method.epsilon}")
        method.numerical_solution_with_many_random_x_0(precise_answer, amount_of_vectors=10)
        print("[END]")
