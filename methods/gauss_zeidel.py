from methods_interface import AnyNumericalMethod
from utils import get_norma_1
import numpy as np


def special_sum_for_critetion(A_matrix, i):
    """
    sum(a[i][j]) if i != j
    """
    result = 0
    for j in range(A_matrix.shape[1]):
        if i == j:
            continue
        result += A_matrix[j][i]
        
    return result


class GaussZeidelMethod(AnyNumericalMethod):

    def __init__(self, A_matrix, b_vector, epsilon, debug = False):
        super().__init__(A_matrix, b_vector, epsilon, debug)
        self.L_matrix = A_matrix.copy()  # Элементы 
        self.L_matrix[np.triu_indices_from(self.L_matrix, k=0)] = 0
        if debug: print("L-matrix = \n", self.L_matrix)

        self.U_matrix = A_matrix.copy()
        self.U_matrix = np.triu(A_matrix)
        np.fill_diagonal(self.U_matrix, 0)
        if debug: print("U-matrix = \n", self.U_matrix)

        self.D_matrix = np.diagflat(np.diag(A_matrix))
        if debug: print("D-matrix = \n", self.D_matrix)


    def check_condition_of_usage(self):
        """
        1. Absolute Eigen vals (L+D)-inv * U less than 1 

        """
        if self.debug: print("[DEBUG] CHECK CONDITIONS")

        L_plus_D_inversed_dotted_U_matrix = np.linalg.inv(self.L_matrix + self.D_matrix).dot(self.U_matrix)
        if self.debug: print("[DEBUG] (L+D)-inv * U = \n", L_plus_D_inversed_dotted_U_matrix)

        abs_eigen_values_checked = map(lambda x: np.absolute(x) < 1, np.linalg.eigvals(L_plus_D_inversed_dotted_U_matrix))
        if self.debug: print("[DEBUG] Is okay to use it?", all(abs_eigen_values_checked))
        return all(abs_eigen_values_checked)
    
    def numerical_solution(self, x_0 = None):
        """
        Easy iteration solution:
        1. Use formula (L+D)-inv * b - (L+D)-inv * U * x_i
        2. Use formula until ||x(k) - x(k+1)|| < epsilon
        """
        super().numerical_solution()
    
        if self.debug: print("[DEBUG] NUMERICAL SOLUTION")
        
        if x_0 is None: 
            x_i = np.array([0] * self.A_matrix.shape[0])
        else:
            x_i = x_0.copy()

        L_plus_D_inv = np.linalg.inv(self.L_matrix + self.D_matrix)
        
        x_next = L_plus_D_inv.dot(self.b_vector) - L_plus_D_inv.dot(self.U_matrix).dot(x_i)

        i = 1
        while abs(sum(x_next - x_i)) >= self.epsilon / 10:
            if self.debug: print(f"[DEBUG] Iter {i} was:", x_i)

            self.past_solutions.save_iteration(x_i)
            x_i = x_next.copy()

            x_next = L_plus_D_inv.dot(self.b_vector) - L_plus_D_inv.dot(self.U_matrix).dot(x_i)
            i += 1
        return x_i
