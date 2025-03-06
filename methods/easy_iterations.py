from methods_interface import AnyNumericalMethod
from utils import get_norma_1
import numpy as np


class EasyIterations(AnyNumericalMethod):

    def __init__(self, A_matrix, b_vector, epsilon, debug = False):
        super().__init__(A_matrix, b_vector, epsilon, debug)
        self.tau = 0

    def check_condition_of_usage(self):
        """
        1. Exists tau, for: || E - tau * A || < 1
        2. If every eigen value for A_transposed * A matrix < 1
        """

        if self.debug: print("[DEBUG] Check conditions of usage")

        delta_t = 0.001  # More accuracy.
        get_norma = get_norma_1

        tau_to_check = 0  # Better from 0 to 1.
        while tau_to_check < 1:
            norma = get_norma(np.eye(*self.A_matrix.shape) - tau_to_check * self.A_matrix, False)
            if norma > 1:
                tau_to_check += delta_t
                continue

            matrix_for_eigen_values = np.eye(*self.A_matrix.shape) - tau_to_check * self.A_matrix
            # if self.debug: print("[DEBUG] What if tau =", tau_to_check)
            # if self.debug: print("[DEBUG] Eigen values: ", np.linalg.eigvals(matrix_for_eigen_values))

            if all(map(lambda x: np.absolute(x) < 1, np.linalg.eigvals(matrix_for_eigen_values))):
                if self.debug: print("[DEBUG] Tau =", tau_to_check)
                if self.debug: print("[DEBUG] Eigen values for check: ", np.linalg.eigvals(matrix_for_eigen_values))
                if self.debug: print("[OK] Method applicable.")
                self.tau = tau_to_check
                return True

            tau_to_check += delta_t
    
    def numerical_solution(self, x_0 = None):
        """
        Easy iteration solution:
        1. Use formula x(k+1) =  (E - tau * A)*x(k) + tau * b
        2. Use formula until ||x(k) - x(k+1)|| < epsilon
        """
        super().numerical_solution()

        if self.debug: print("[DEBUG] NUMERICAL SOLUTION")
        zero_matrix = np.eye(*self.A_matrix.shape)

        if x_0 is None: 
            x_i = np.array([0] * self.A_matrix.shape[0])
        else:
            x_i = x_0.copy()

        x_next = self.tau * np.transpose(self.b_vector)

        if self.debug: print("[DEBUG] x_0 =", x_i)
        if self.debug: print("[DEBUG] x_1 =", x_next)

        i = 1
        while abs(sum(x_next - x_i)) > self.epsilon:
            if self.debug: print(f"[DEBUG] Iter {i} was:", x_i)
            self.past_solutions.save_iteration(x_i)

            x_i = x_next.copy()
            x_next = (zero_matrix - self.tau * self.A_matrix).dot(x_i) + self.tau * np.transpose(self.b_vector)
            i += 1

        return x_i
