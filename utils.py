import numpy as np


def get_norma_1(A_matrix, debug=True):
    """
    Расчёт нормы по формуле
    square(max(abs(lambda_i)))
    """
    if debug: print("[DEBUG] GET NORMA 1 for A =\n", A_matrix)
    if debug: print("[DEBUG] Formula: square{max [abs(lambda_i)]}")
    eigen_values = np.linalg.eigvals(np.transpose(A_matrix) * A_matrix)
    abs_eigen_values = list(map(lambda x: np.absolute(x), eigen_values))
    if debug: print("[DEBUG] Eigien values(abs) =", list(map(lambda x: float(np.absolute(x)), eigen_values)))
    if debug: print("[DEBUG] norma_1 =", max(abs_eigen_values) ** 0.5)

    return max(abs_eigen_values) ** 0.5


def get_norma_2(A_matrix, debug=True):
    """
    Расчёт нормы по формуле
    max{sum[ abs( a[i][j] ) ]}
    Суммирует по строке!
    """
    if debug: print("[DEBUG] GET NORMA 2 for A =\n", A_matrix)
    if debug: print("[DEBUG] Formula: max{sum[ abs( a[i][j] ) ]}")
    matrix_norms_2 = list()
    for line in range(A_matrix.shape[0]):
        summary = 0
        for row in range(A_matrix.shape[1]):
            summary += np.absolute(A_matrix[line][row])
            if debug: print((line, row, int(A_matrix[line][row])), end="\t")
        if debug: print("[DEBUG] Summary for {line =", line, "} = ", summary)
        matrix_norms_2.append(summary)
    if debug: print("[DEBUG] norma_2 =", max(matrix_norms_2))
    
    return max(matrix_norms_2)


def get_norma_3(A_matrix, debug=True):
    """
    Расчёт нормы по формуле
    max{sum[ abs( a[i][j] ) ]}
    Суммирует по столбцам!
    """
    matrix_norms_3 = list()
    if debug: print("Matrix below is transposed, mind that:")
    for row in range(A_matrix.shape[0]):
        summary = 0
        for line in range(A_matrix.shape[1]):
            summary += np.absolute(A_matrix[line][row])
            if debug: print((line, row, int(A_matrix[line][row])), end="\t")
        if debug: print("Summary for {row =", row, "} = ", summary)
        matrix_norms_3.append(summary)

    if debug: print("[DEBUG] norma_3 =", max(matrix_norms_3))
    return max(matrix_norms_3)  # Here we gooo...
