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
        3931 / 4336, 3357 / 2168, -547 / 1084, 150 / 271
    ]
)


methods = {
    "EasyIterations": [
        # Список аргументов: матрица системы, вектор свободных членов, epsilon (точность), debug-режим
        EasyIterations(A_matrix, b_vector, 10**(-1), False),
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
        GaussZeidelMethod(A_matrix, b_vector, 10**(-3), False),
    ]
}


for method_name, methods_realizations in methods.items():
    method: AnyNumericalMethod
    for method in methods_realizations:
        print("[START]", "DEBUG MODE" if method.debug else "INFO MODE")
        is_ok = method.base_check_conditions()
        if not is_ok:
            print("[INFO] VERDICT --> IT IS NOT OK BASICALLY, SKIP")
            continue

        is_ok = method.check_condition_of_usage()
        if not is_ok:
            print("[INFO] VERDICT --> IT IS NOT OK, check criteria, SKIP")
            continue

        print("[INFO] Condition numbers")
        mu_1, mu_2, mu_3 = method.count_condition_numbers()

        print(f"[INFO] mu_1 = {mu_1}")
        print(f"[INFO] mu_2 = {mu_2}")
        print(f"[INFO] mu_3 = {mu_3}")

        print(f"[INFO] CHECK GRAPHICS. METHOD __{method_name}__; PRESISION = {method.epsilon}")
        # Creating random vectors and drawing the graphics N(S).
        # Список аргументов: точный вектор-ответ, количество рандомных векторов, показывать ВЕСЬ отладочный процесс.
        method.numerical_solution_with_many_random_x_0(precise_answer, amount_of_vectors=30, show_all_debug=True)
        print("[END]")
