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


class JacobiMethod(AnyNumericalMethod):

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

    
    def __check_jacobi_criterion__(self):
        """
        Absolute ALL Eigen values (of matr. (D-inv * (L + U)) ) less than 1
        """
        matrix = np.linalg.inv(self.D_matrix).dot(self.L_matrix + self.U_matrix)
        if self.debug: print("[DEBUG] D-inv = \n", matrix)
        if self.debug: print("[DEBUG] Eigen vals for D-inv * (L + U) =\n", np.linalg.eigvals(matrix))
        abs_eigen_values_less_1 = map(lambda x: np.absolute(x) < 1, np.linalg.eigvals(matrix))
        if self.debug: print("[DEBUG] Is ok to use this method (Yes = True; No = False)?", all(abs_eigen_values_less_1))
        return all(abs_eigen_values_less_1)


    def check_condition_of_usage(self):
        """
        1. Diagonal dominance:
        |a[i][i]| > sum(a[i][j]) ( i != j )
        2. Jacobi criterion
        Absolute ALL Eigen values (of matr. (D-inv * (L + U)) ) less than 1. 

        """
        if self.debug: print("[DEBUG] ")
        for i in range(self.A_matrix.shape[0]):
            if self.debug: print(f"[DEBUG] i={i}, a[i][i]={self.A_matrix[i][i]} sum={special_sum_for_critetion(self.A_matrix, i)}")
            if np.abs(self.A_matrix[i][i]) > special_sum_for_critetion(self.A_matrix, i):
                if self.debug: print(f"[DEBUG] Element under A[{i}][{i}] > {special_sum_for_critetion(self.A_matrix, i)}")
                if self.debug: print(f"[DEBUG] Let us check Jacobi criterion...")
                return self.__check_jacobi_criterion__()
        return False
    
    def numerical_solution(self, x_0 = None):
        """
        Easy iteration solution:
        1. Use formula (check ReadMe.md)
        2. Use formula until ||x(k) - x(k+1)|| < epsilon
        """
        super().numerical_solution()

        if self.debug: print("[DEBUG] NUMERICAL SOLUTION")
        D_inv = np.linalg.inv(self.D_matrix)

        if x_0 is None: 
            x_i = np.array([0] * self.A_matrix.shape[0])
        else:
            x_i = x_0.copy()


        x_next = D_inv.dot(self.b_vector - (self.L_matrix + self.U_matrix).dot(x_i))

        i = 1
        while abs(sum(x_next - x_i)) > self.epsilon:
            if self.debug: print(f"[DEBUG] Iter {i} was:", x_i)
            self.past_solutions.save_iteration(x_i)
            
            x_i = x_next.copy()
            x_next = D_inv.dot(self.b_vector - (self.L_matrix + self.U_matrix).dot(x_i))
            i += 1
        return x_i
