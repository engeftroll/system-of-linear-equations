from typing import Tuple
import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SolutionMemoryForLinearSystem:
    def __init__(self, path_to_save: str = "./"):
        self.solutions_list = list()  # Inside there is a LIST of NORMALS!
        self.path_to_save_information = path_to_save
    
    def get_dictionary(self):
        return {
            "solutions": self.solutions_list,
            "precision_relatively_last_solution": [self.get_norma(x, self.solutions_list[-1]) for x in self.solutions_list],
            "precision_relatively_zero_solution": [self.get_norma(x, self.solutions_list[0]) for x in self.solutions_list],
            "iteration_num": [i + 1 for i in range(len(self.solutions_list))]
        }

    def save_iteration(self, x_i):
        self.solutions_list.append(x_i)

    def clear_history(self):
        self.solutions_list.clear()

    @staticmethod
    def get_norma(x_i, x_precised):
        # return max(x_i - x_precised)
        # return sum(abs(x_i - x_precised))
        return (sum(map(lambda x: x ** 2, x_i - x_precised))) ** 0.5
    

    def show_plot(self):
        data_frame = pd.DataFrame(self.get_dictionary())
        data_frame.plot(title="Precise", x="iteration_num", y="precision_relatively_zero_solution")
        plt.show()


class AnyNumericalMethod:
    def __init__(self, A_matrix, b_vector, epsilon, debug: bool = False):
        """
        A_matrix: матрица СЛАУ
        b_vector: вектор b
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
        if self.debug: print("[DEBUG] COUNT CONDITION NUMBERS")
        A_matrix_inversed = np.linalg.inv(self.A_matrix)

        if self.debug: print("[DEBUG] A-matrix inversed = \n", A_matrix_inversed)

        if self.debug: print("[DEBUG] Count via norma_1: ")
        mu_1 = utils.get_norma_1(A_matrix_inversed, self.debug) / utils.get_norma_1(self.A_matrix, self.debug)
        if self.debug: print("[DEBUG] mu_1 =", mu_1)

        if self.debug: print("[DEBUG] Count via norma_2: ")
        mu_2 = utils.get_norma_2(A_matrix_inversed, self.debug) / utils.get_norma_2(self.A_matrix, self.debug)
        if self.debug: print("[DEBUG] mu_2 =", mu_2)

        if self.debug: print("[DEBUG] Count via norma_3: ")
        mu_3 = utils.get_norma_3(A_matrix_inversed, self.debug) / utils.get_norma_3(self.A_matrix, self.debug)
        if self.debug: print("[DEBUG] mu_3 =", mu_3)

        return (float(mu_1), float(mu_2), float(mu_3))

    def check_condition_of_usage(self) -> bool:
        """Критерии сходимости"""
        pass
    
    def numerical_solution(self):
        """Расчёт до заданной точности epsilon. Не забывай сохранять промежуточные этапы!"""
        self.past_solutions.clear_history()
