import math
import random
import scipy
import numpy as np


def get_input_from_file():
    """
    Function for reading input from input file text.
    :return:
    """
    n, eps, M, b = None, None, [], []
    with open('in.txt', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # 1st line: n
                n = line
            elif i == 1:  # 2nd line: eps
                eps = line
            else:
                M.append([float(x) for x in line.split(' ')])
                A = np.asarray(M)
                print(A[2, 2])
    return n, eps, A, b


def generate_random_matrix(n):
    """
    Function for generating a matrix A with elements randomly chosen in (0, 10000).

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


def get_transpose(A):
    """
    Function that returns the transpose of a matrix A.

    :param A: matrix A
    :return: transpose of A
    """
    A_T = np.zeros(n)
    for row in range(0, n):
        for col in range(0, n):
            A_T[row][col] = A[col][row]
    return A_T


def check_matrix_symmetry(A) -> bool:
    """
    Boolean function determining whether a matrix A is symmetric (i.e., A is equal to its transpose).

    :param A: matrix A
    :return: boolean value (true if At = A, false otherwise)
    """
    for row in range(0, n):
        for col in range(row, n):
            if A[row][col] != A[col][row]:
                return False
    return True


def get_A_init(A):
    """

    :param A: initial matrix A, before it undergoes any changes
    :return: copy of matrix A
    """
    A_initial = np.copy(A)
    return A_initial


# det(A) = det(L)det(Lt) = l11**2 * l22**2 * ... lnn**2
def LUDet(A):
    p = 1
    for index in range(0, n):
        p *= A[index][index] ** 2
    return p


def check_diagonal_is_positive(A) -> bool:
    """
    Boolean function determining whether a matrix A has exclusively positive elements on its main diagonal.
    :param A: matrix A
    :return: true if l(i,i) > 0 for all i, false otherwise
    """
    for index in range(0, n):
        if A[index][index] <= 0:
            return False
    return True


def cholesky_decomposition(A):
    """
    Function for computing the Cholesky decomposition of a matrix A (as explained online).

    :param A: matrix for which we perform the Cholesky decomposition
    :return: L and transpose L (lower and upper triangular matrix)
    """
    # L = scipy.linalg.cholesky(A, lower=True)     # lower triangular matrix in Cholensky decomposition for A:
    # L_t = scipy.linalg.cholesky(A, lower=False)  # upper ~
    # return L, L_t

    # lower triangular matrix in Cholesky decomposition for A:
    L = np.zeros(n)
    for row in range(0, n):
        for col in range(0, row + 1):
            s = 0
            if row == col:  # main diagonal
                for p in range(0, col):
                    s += L[row][p] ** 2
                L[col][col] = math.sqrt(A[col][col] - s)
            else:
                for p in range(0, col):
                    s += L[row][p] * L[col][p]
                if L[col][col] > 0:
                    L[col][col] = (A[row][col] - s) / L[col][col]
    if check_diagonal_is_positive(L) is True:
        return L, get_transpose(L)
    return "Error! Cholesky decomposition not possible."


def cholesky(A):
    """
    Function for computing the Cholesky decomposition of a matrix A (as given in homework pdf).

    :param A: matrix for which we perform the Cholensky decomposition
    :return: A as
    """
    # L = scipy.linalg.cholesky(A, lower=True)     # lower triangular matrix in Cholensky decomposition for A:
    # L_t = scipy.linalg.cholesky(A, lower=False)  # upper ~
    # return L, L_t

    possible = True
    for p in range(0, n):
        for row in range(p, n):
            upper = 0
            lower = 0
            for col in range(0, p):
                upper += A[p][col] * A[col][row]
            A[p][row] -= upper
            if row == p:
                pass
            else:
                for col in range(0, p):
                    lower = A[row][col] * A[col][p]
                if abs(A[p][p] > eps):
                    A[row][p] = (A[row][p] - lower) / A[p][p]
                else:
                    possible = False
                    break
    if possible is True and check_diagonal_is_positive(A):
        return A
    print("Division by 0!")
    return possible


"""
L * y   = b  --> first
L_t * x = y  --> second
"""


def check_cholesky_is_correct(A) -> bool:
    """
    Boolean function for checking whether our Cholesky decomposition is correct.
    A correct Cholesky decomposition of A = L * L_t will hold that L * L_t = A_init.

    :param A: matrix A
    :return: true if matrix multiplication is same as A_init, false otherwise
    """
    A_init = get_A_init(A)
    L = cholesky(A)
    L_t = get_transpose(L)
    product = np.matmul(L, L_t)
    for row in range(0, n):
        for col in range(0, n):
            if product[row][col] != A_init[row][col]:
                return False
    return True


def solve_Ly_equals_b():
    """
    Direct substitution.

    :param b:
    :return:
    """
    Y = []
    possible = True
    for row in range(0, n):
        y = b[row]
        for col in range(0, row):
            y -= L[row][col] * Y[col]
        if abs(L[row][row]) <= eps:
            print("Error! Can't perform division.")
            possible = False
        else:
            Y.append(y / L[row][row])
    if possible is True:
        return Y


def solve_Lt_x_equals_y_star(Y):
    """
    Inverse substitution.
    :param x:
    :return:
    """
    L_t = get_transpose(L)
    X = np.zeros(n)
    for row in range(n-1, 0, -1):
        x = Y[row]
        for col in range(row + 1, n):
            x -= X[col] * L_t[row][col]
        if abs(L_t[row][row]) <= eps:
            break
        else:
            X[row] = x / L_t[row][row]
    return X


def solve_system(L, b):

    Y = solve_Ly_equals_b()
    X = solve_Lt_x_equals_y_star(Y)
    return X


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
    copy_A = np.copy(A)
    A_inverse = np.zeros(n, dtype=float)
    for index in range(0, n):
        # getting vector b as a vector with 0s and a 1 on the position 'index'
        b = np.zeros(n)
        b[index] = 1.0
        # for each column in matrix A, we solve the system
        # first: solving L * y = b:
        
    return 0




def get_inverse_by_lib(A):
    """
    Function that returns the inverse matrix of matrix A as computed by python's library.

    :param A: matrix A
    :return: inverse of A
    """
    return np.linalg.inv(A)


def compute_norm_for_LU_decomposition(A_init, b):
    """
    Function for computing the Euclidean norm ||A_init * x_cholesky - b||2 with the Python library
    (used for checking rate of success for our computation).
    For reference we will use epsilon = 10 ** (-9)

    :return: euclidean norm value
    """
    X_chol = np.linalg.solve(A_init, b)     # computing x_cholesky
    mul = np.dot(A_init, X_chol)            # A_init * X_cholesky
    mul = np.subtract(mul, b)               # subtracting b
    val = np.linalg.norm(mul)               # computing euclidean norm on new result
    if val < pow(10, -9):
        print("Norm ||A_init * x_cholesky - b||2 is " + val + ". Computation is correct with the error epsilon.")
    else:
        print("Norm ||A_init * x_cholesky - b||2 is " + val + ". Computation is faulty.")
    print('\n')
    return val


def compute_norm_for_approximation_of_cholesky_inverse(A_cholesky):
    """
    Function for computing the approximation for the inverse of A_cholesky
    ||A(-1, cholesky) - A(-1, bibl)||

    :return: euclidean norm value
    """
    A_inverse_chol = get_inverse_by_hand(A_cholesky)            # inverse computed by us
    A_inverse_python = np.linalg.inv(np.matrix(A_cholesky))     # inverse computed by python library
    sub = np.subtract(A_inverse_chol, A_inverse_python)         # A(-1, cholesky) - A(-1, bibl)
    return np.linalg.norm(sub)                                  # computing norm on previously obtain result


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

        m = random.randint(5, 14)  # eps = 10 ** (-m), m in [5,13]
        eps = 10 ** (-m)
        n = random.randint(3, 201)  # matrix size in [3, 200]
        M = generate_random_matrix(n)
        M_init = get_A_init(M)
        b = []
        for index in range(0, n):
            b[index] = random.uniform(1.0, 1000.0)
        print("n =", n)
        print("eps =", eps)
        print("M=", M)
        print("b =", b)

    elif input_method == 'b':
        # if input data is taken from file:
        n, eps, M, b = get_input_from_file()

    else:
        # if input data is given from console:
        pass


if __name__ == "__main__":
    main()







