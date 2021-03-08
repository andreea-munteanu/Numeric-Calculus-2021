import math
import random
import scipy
from scipy import linalg
import numpy as np


def get_input_from_file(n):
    """
    Function for reading input from input file text.
    :return:
    """
    A = []
    b = []
    with open('in.txt', 'r') as file:
        for i, line in enumerate(file):
            M = []
            if i < n:
                M.append([float(x) for x in line.split(' ')])
                A.append(M)
            else:
                b.append([float(x) for x in line.split(' ')])
    return A, b


# A, b = get_input_from_file(3)
# print(A)
# print("b=", b)


def generate_random_matrix(n):
    """
    Function for generating a symmetric matrix A with elements randomly chosen in (0, 10000).

    :param n: size of square matrix A
    :return: matrix A randomly generated
    """
    A = np.zeros(n)
    for row in range(0, n):
        for col in range(row, n):
            A[row][col] = A[col][row] = random.uniform(0, 10001)
    return A


m = 8
n = 3
eps = pow(10, -m)
M = np.matrix([[1.0, 0.0, 3.0],
               [0.0, 4.0, 2.0],
               [3.0, 2.0, 11.0]])
b = [1.0, 2.0, 4.0]
d = np.diag(M)  # vector of matrix main diagonal

print("\nM is:\n", M, '\n')
M_init = np.copy(M)


def get_transpose(A):
    """
    Function that returns the transpose of a matrix A.

    :param A: matrix A
    :return: transpose of A
    """
    return A.transpose()


M_T = get_transpose(M)
print("Transpose of M is: \n", M_T)


def check_matrix_symmetry(A) -> bool:
    """
    Boolean function determining whether a matrix A is symmetric (i.e., A is equal to its transpose).

    :param A: matrix A
    :return: boolean value (true if At = A, false otherwise)
    """
    # symmetric = True
    # for row in range(0, n):
    #     for col in range(row, n):
    #         if A[row][col] != A[col][row]:
    #             symmetric = False
    #             break
    # return symmetric
    return (A == get_transpose(A)).all()


print("\nIs our matrix symmetric?", "Yes." if check_matrix_symmetry(M) else "No.")


def get_A_init(A):
    """

    :param A: initial matrix A, before it undergoes any changes
    :return: copy of matrix A
    """
    A_initial = np.copy(A)
    return A_initial


def check_diagonal_is_positive(A) -> bool:
    """
    Boolean function determining whether a matrix A has exclusively positive elements on its main diagonal.

    :param A: matrix A
    :return: true if l(i,i) > 0 for all i, false otherwise
    """
    return (A >= 0.0).all()


print("\nIs our matrix's diagonal positive? ", "Yes." if check_diagonal_is_positive(M) else "No.")


def library_cholesky_decomposition(A):
    """
    Function for getting the L, L_t Cholesky decomposition as computed by the numpy python library.

    :param A: matrix
    :return: L, L_t
    """
    L = scipy.linalg.cholesky(A, lower=True)     # lower triangular matrix in Cholensky decomposition for A:
    L_t = scipy.linalg.cholesky(A, lower=False)  # upper ~
    return L, L_t


L, L_t = library_cholesky_decomposition(M)
print("\nCholesky decomposition (numpy): \n", L, '\n\n', L_t, '\n')


def cholesky_decomposition(A):
    """
    Function for computing the Cholesky decomposition of a matrix A.

    :param A: matrix for which we perform the Cholesky decomposition
    :return: L and transpose L (lower and upper triangular matrix)
    """

    # lower triangular matrix in Cholesky decomposition for A:
    L = np.copy(A)
    # first element of matrix:
    L[0, 0] = A[0, 0]
    # # first col of matrix:
    # for row in range(1, n):
    #     L[row, 0] = A[0, row] / L[0, 0]
    for row in range(0, n):
        for col in range(0, row + 1):
            s = 0
            if row == col:  # main diagonal
                for p in range(0, col):
                    s += L[row, p] ** 2
                L[col, col] = math.sqrt(A[col, col] - s)
            else:
                for p in range(0, col):
                    s += L[row, p] * L[col, p]
                if abs(L[col, col]) > eps:
                    L[col, col] = (A[row, col] - s) / L[col, col]
                    print(L[col, col])
                else:
                    print("Division by 0")
                    break

    if check_diagonal_is_positive(L) is True:
        # return lower and upper triangular matrix:
        return L, get_transpose(L)
    # print("\tError! Cholesky decomposition not possible.")
    return None, None


# L, L_t = cholesky_decomposition(M)
# print("\nCholesky decomposition: \n", L, '\n\n', L_t, '\n')


def get_diagonal(matrix):
    """
    :param matrix:
    :return: array containing elements on main diagonal of matrix A
    """
    return matrix.diagonal()


def compute_det_A(L, L_t):
    """
    Function for computing det(A) = det(L) * det(L_t) = (l11 * l22 * .. * lnn) * (l11 * l22 * .. * lnn)

    :param L: lower triangular matrix
    :param L_t: upper triangular matrix
    :return: det(A)
    """
    diag_L = get_diagonal(L)
    diag_L_t = get_diagonal(L_t)
    det_A = 1
    for index in range(0, n):
        det_A = det_A * diag_L[index] * diag_L_t[index]
    return det_A


print("\ndet(M) = ", compute_det_A(L, L_t))


def check_cholesky_is_correct(A) -> bool:
    """
    Boolean function for checking whether our Cholesky decomposition is correct.
    A correct Cholesky decomposition of A = L * L_t will hold that L * L_t = A_init.

    :param A: matrix A
    :return: true if matrix multiplication is same as A_init, false otherwise
    """
    A_init = get_A_init(A)
    L, L_t = library_cholesky_decomposition(A)  # cholesky_decomposition(A)
    product = np.matmul(L, L_t)
    for row in range(0, n):
        for col in range(0, n):
            if product[row, col] != A_init[row, col]:
                return False
    return True


print("\nIs our Cholesky decomposition correct? ", "Yes." if check_cholesky_is_correct(M) else "No.")


"""
L * y   = b  --> first
L_t * x = y  --> second
"""


def solve_Ly_equals_b(L, b): # checked; works
    """
    Solve L * y = b  =>  find y*

    :param b: initial vector b
    :return:
    """
    Y = []
    possible = True
    for row in range(0, n):
        y = b[row]
        for col in range(0, row):
            y -= L[row, col] * Y[col]
        if abs(L[row, row]) < eps:
            print("Error! Can't perform division.")
            possible = False
        else:  # if abs(L[row, row]) > eps: can perform division
            Y.append(y / L[row, row])
    if possible is True:
        return Y


def solve_Lt_x_equals_y_star(L, Y): # checked; works
    """
    Solve L_t * x = Y
    """
    L_t = get_transpose(L)
    X = np.zeros(n)
    for row in range(0, n):
        x = Y[row]
        for col in range(row + 1, n):
            x -= X[col] * L_t[row][col]
        if abs(L_t[row][row]) <= eps:
            break
        else:
            X[row] = x / L_t[row][row]
    return X


def solve_system(A, b):
    L, L_t = library_cholesky_decomposition(A)  # cholesky_decomposition(A)
    Y = solve_Ly_equals_b(L, b)
    X = solve_Lt_x_equals_y_star(L_t, Y)
    return X


print("\nX_cholesky is: ", solve_system(M, b))


def get_inverse_by_hand(A):
    """
    Function that returns the inverse matrix of matrix A.
    For computing column j = 1..n of A_transposed, we will solve A * x = l(j):
    l(1) = (1, 0, 0, ..., 0)
    l(2) = (0, 1, 0, ..., 0)
    ...
    b = l(i)  =>  We will have to solve A * x == l(i)  <=>  1) L * y = b(i) and we find y
                                                            2) l_t * x = y* and we find x
    :param A: matrix A
    :return: inverse of A
    """
    A_inverse = np.zeros([n, n], dtype=float)
    for index in range(0, n):
        # getting vector b as a vector with 0s and a 1 on the position 'index'
        b = np.zeros(n)
        b[index] = 1.0
        # for each column in matrix A, we solve the system and the solution X will be col_i of A_inverse
        col_i = solve_system(A, b)
        for row in range(0, n):
            A_inverse[row, index] = col_i[row]
    return A_inverse


print("\n Matrix M's inverse matrix: \n", get_inverse_by_hand(M))


def get_inverse_by_lib(A):
    """
    Function that returns the inverse matrix of matrix A as computed by python's library.

    :param A: matrix A
    :return: inverse of A
    """
    return np.linalg.inv(A)


# print("\n Matrix M's inverse matrix (numpy): \n", get_inverse_by_lib(M))


def compute_norm_for_LLt_decomposition(A_init, b):
    """
    Function for computing the Euclidean norm ||A_init * x_cholesky - b||2 with the Python library
    (used for checking rate of success for our computation).
    For reference we will use epsilon = 10 ** (-9)

    :return: euclidean norm value
    """
    X_chol = solve_system(A_init, b)
    # X_chol = np.linalg.solve(A_init, b)     # computing x_cholesky
    mul = np.dot(A_init, X_chol)            # A_init * X_cholesky
    mul = np.subtract(mul, b)               # subtracting b
    val = np.linalg.norm(mul)               # computing euclidean norm on new result
    if val < pow(10, -9):
        print("\nNorm ||A_init * x_cholesky - b|| is ", val, ". \nComputation is correct with the error epsilon.")
    else:
        print("\nNorm ||A_init * x_cholesky - b|| is ",  val,  ". \nComputation is faulty.")


compute_norm_for_LLt_decomposition(M_init, b)


print("\n Matrix M's inverse matrix: \n", get_inverse_by_hand(M))


def compute_norm_for_approximation_of_cholesky_inverse(A_cholesky):
    """
    Function for computing the approximation for the inverse of A_cholesky
    ||A(-1, cholesky) - A(-1, bibl)||

    :return: euclidean norm value
    """
    A_inverse_chol = get_inverse_by_hand(A_cholesky)            # inverse computed by us
    A_inverse_python = np.linalg.inv(np.matrix(A_cholesky))     # inverse computed by python library
    sub = np.subtract(A_inverse_chol, A_inverse_python)         # A(-1, cholesky) - A(-1, bibl)
    return np.linalg.norm(sub)                                  # computing norm on previously obtain results


print("\nNorm ||A(-1, cholesky) - A(-1, bibl)||: ", compute_norm_for_approximation_of_cholesky_inverse(M))


def main():
    print("Input method:\na) Randomized\nb) From file\nc) From console\nInsert your choice (a, b or c): ", end='')
    correct_input: bool = False
    input_method = None
    while not correct_input:
        input_method = input()
        if input_method in "abc":
            correct_input = True
    # print("Chosen method: " + input_method + '\n')
    if input_method == 'a':
        # if input data is chosen pseudo-randomly:

        m = random.randint(5, 14)        # eps = 10 ** (-m), m in [5,13]
        eps = 10 ** (-m)
        n = random.randint(3, 201)       # matrix size in [3, 200]
        M = generate_random_matrix(n)    # symmetrical already, no need to further check
        M_init = get_A_init(M)           # A_init
        b = []
        for index in range(0, n):
            b[index] = random.uniform(1.0, 1000.0)
        print("n =", n)
        print("eps =", eps)
        print("M =", M)
        print("b =", b)

        M_T = get_transpose(M)
        print("Transpose of M is: \n", M_T)
        print("\nIs our matrix symmetric?", "Yes." if check_matrix_symmetry(M) else "No.")
        print("\nIs our matrix's diagonal positive? ", "Yes." if check_diagonal_is_positive(M) else "No.")
        L, L_t = library_cholesky_decomposition(M)
        print("\nCholesky decomposition (numpy): \n", L, '\n\n', L_t, '\n')
        print("\ndet(M) = ", compute_det_A(L, L_t))
        print("\nIs our Cholesky decomposition correct? ", "Yes." if check_cholesky_is_correct(M) else "No.")
        print("\nX_cholesky is: ", solve_system(M, b))
        print("\n Matrix M's inverse matrix: \n", get_inverse_by_hand(M))
        compute_norm_for_LLt_decomposition(M_init, b)

    elif input_method == 'b':
        # if input data is taken from file:
        n = 3
        m = 8
        eps = 10 ** (-m)
        M, b = get_input_from_file(n)
        M_init = np.copy(M)
        M_T = get_transpose(M)
        print("Transpose of M is: \n", M_T)
        print("\nIs our matrix symmetric?", "Yes." if check_matrix_symmetry(M) else "No.")
        print("\nIs our matrix's diagonal positive? ", "Yes." if check_diagonal_is_positive(M) else "No.")
        L, L_t = library_cholesky_decomposition(M)
        print("\nCholesky decomposition (numpy): \n", L, '\n\n', L_t, '\n')
        print("\ndet(M) = ", compute_det_A(L, L_t))
        print("\nIs our Cholesky decomposition correct? ", "Yes." if check_cholesky_is_correct(M) else "No.")
        print("\nX_cholesky is: ", solve_system(M, b))
        print("\n Matrix M's inverse matrix: \n", get_inverse_by_hand(M))
        compute_norm_for_LLt_decomposition(M_init, b)

    else:
        # if input data is given from console:
        print("n = ", end='')
        n = input()
        print("m = ", end='')
        m = input()
        b = []
        bv = str(input("b = "))
        bv.split(" ")
        for i in b:
            b.append(float(i))

