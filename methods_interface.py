from typing import Tuple
import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import randint


class SolutionMemoryForLinearSystem:
    """
    Класс, который хранит информацию о работе одного численного метода, 
    с единственным вариантом приближения
    """

    def __init__(self, path_to_save: str = "./"):
        self.solutions_list = list()  # Inside there is a LIST of NORMALS!
    
    def get_dictionary(self):
        return {
            "solutions": self.solutions_list,
            "precision_relatively_last_solution": [self.get_norma(x - self.solutions_list[-1]) for x in self.solutions_list],
            "precision_relatively_zero_solution": [self.get_norma(x - self.solutions_list[0]) for x in self.solutions_list],
            "iteration_num": [i + 1 for i in range(len(self.solutions_list))]
        }

    def save_iteration(self, x_i):
        self.solutions_list.append(x_i)

    def clear_history(self):
        self.solutions_list.clear()

    @staticmethod
    def get_norma(x_i):
        """Получение нормы вектора (евклидова норма)"""
        # return max(x_i)
        # return sum(abs(x_i))
        return (sum(map(lambda x: x ** 2, x_i))) ** 0.5
    
    def get_result(self):
        """Получение последнего расчёта"""
        return self.solutions_list[-1]
    

    def show_plot(self):
        """Показать график N(|x0-x_i|). x0 - начальное приближение, x_i - i-ое приближение"""
        data_frame = pd.DataFrame(self.get_dictionary())
        data_frame.plot(title="Precise", x="iteration_num", y="precision_relatively_zero_solution")
        plt.show()


class AnyNumericalMethod:
    """Представление любого численного метода (родитель, от которого все методы наследуются)"""
    def __init__(self, A_matrix, b_vector, epsilon, debug: bool = False):
        """
        A_matrix: матрица СЛАУ
        b_vector: вектор b
        epsilon: точность системы
        debug: режим отладки (вывод всех действий с матрицами/векторами)
        """
        self.A_matrix = A_matrix
        self.b_vector = b_vector
        self.debug = debug
        self.epsilon = epsilon

        self.past_solutions = SolutionMemoryForLinearSystem()

    def base_check_conditions(self) -> bool:
        """Проверка, возможно ли вообще решить такую СЛАУ"""

        lines_amount, elements_amount = self.A_matrix.shape
        if self.debug: print("[DEBUG] Base check conditions")
        if self.debug: print("[DEBUG] Are amount of rows == amount of lines?")
        if lines_amount != elements_amount:
            print("[ERR] No way to resolve this system definetely.")
            return False
        if self.debug: print("[DEBUG] OK")
        

        determinant = np.linalg.det(self.A_matrix)
        if self.debug: print("[DEBUG] Determinant(A) =", determinant, " != 0?")
        if determinant == 0:
            print("[ERR] Determinant of A-matrix = 0, make system easier...")
            return False
        if self.debug: print("[DEBUG] OK")
        if self.debug: print("[DEBUG] Base check conditions OK")
        return True


    def count_condition_numbers(self) -> Tuple[float, float, float]:
        """Расчёт трёх чисел обусловленности"""
        if self.debug: print("[DEBUG] COUNT CONDITION NUMBERS\n")
        A_matrix_inversed = np.linalg.inv(self.A_matrix)

        if self.debug: print("[DEBUG] A-matrix inversed = \n", A_matrix_inversed)

        if self.debug: print("[DEBUG] Count via norma_1: ")
        mu_1 = utils.get_norma_1(A_matrix_inversed, self.debug) / utils.get_norma_1(self.A_matrix, self.debug)
        if self.debug: print("\n[DEBUG] mu_1 =", mu_1, "\n")

        if self.debug: print("[DEBUG] Count via norma_2: ")
        mu_2 = utils.get_norma_2(A_matrix_inversed, self.debug) / utils.get_norma_2(self.A_matrix, self.debug)
        if self.debug: print("\n[DEBUG] mu_2 =", mu_2, "\n")

        if self.debug: print("[DEBUG] Count via norma_3: ")
        mu_3 = utils.get_norma_3(A_matrix_inversed, self.debug) / utils.get_norma_3(self.A_matrix, self.debug)
        if self.debug: print("\n[DEBUG] mu_3 =", mu_3, "\n")

        return (float(mu_1), float(mu_2), float(mu_3))

    def check_condition_of_usage(self) -> bool:
        """Критерии сходимости"""
        pass
    
    def numerical_solution(self, x_0 = None):
        """
            x_0: Первое приближение, по умолчанию -- нулевой вектор (np.array!).
            Расчёт до заданной точности epsilon. 
            Не забывай сохранять промежуточные этапы
        """
        self.past_solutions.clear_history()
    
    def x_0_random(self, from_value: int = -10, to_value: int = 10):
        """Создание рандомного вектора x_0 (первого приближения к решению)"""
        return np.array([randint(from_value, to_value) for _ in range(self.A_matrix.shape[0])])
    
    def numerical_solution_with_many_random_x_0(self, precise_vector, amount_of_vectors: int = 10, show_all_debug: bool = False):
        """
        Решение СЛАУ, используя множество различных псевдо-случайных векторов.
        Выхлоп - график, на котором отображается зависимость N(S), где
        N - это количество итераций
        S - это норма разности точного решения (precise_vector) и вектора начального приближения (x_0)
        """

        result = {
            "N": list(),
            "S": list()
        }
        need_debug = show_all_debug
        
        # Creating random vector, where every coordinate from value to value
        random_vectors = sorted(
            [
                self.x_0_random(i * 3, i * 3)  
                for i in range(1, amount_of_vectors)
            ], 
            key=lambda x: -SolutionMemoryForLinearSystem.get_norma(x - precise_vector)
        )

        # Stress test
        # random_vectors = [
        #     np.array([3931 / 4336, 3357 / 2168, -547 / 1084, 150 / 271])
        # ]

        for x_0 in random_vectors:
            self.numerical_solution(x_0=x_0)
            n = len(self.past_solutions.solutions_list)

            if need_debug: print("[DEBUG] x_0 = ", x_0, "\t|x_0 - x*| =", SolutionMemoryForLinearSystem.get_norma(x_0 - precise_vector), f"(N = {n})")
            result["N"].append(n)
            result["S"].append(SolutionMemoryForLinearSystem.get_norma(precise_vector - x_0))

        data_frame = pd.DataFrame(result)
        data_frame.plot(title="N(S)", x="S", y="N")
        plt.show()
