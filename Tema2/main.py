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


def generate_random_matrix(n): # checked; works
    """
    Function for generating a symmetric matrix A with elements randomly chosen in (0, 10000).

    :param n: size of square matrix A
    :return: matrix A randomly generated
    """
    return np.random.random((n, n))


def get_diagonal(matrix):
    """
    :param matrix:
    :return: array containing elements on main diagonal of matrix A
    """
    return np.diagonal(matrix)


m = 8
n = 3
eps = pow(10, -m)
M = np.matrix([[1.0, 0.0, 3.0],
               [0.0, 4.0, 2.0],
               [3.0, 2.0, 11.0]])
b = [1.0, 2.0, 4.0]
d = get_diagonal(M)  # vector of matrix main diagonal

print("\nM is:\n", M)
print("\nd =", d)


def get_transpose(A):
    """
    Function that returns the transpose of a matrix A.

    :param A: matrix A
    :return: transpose of A
    """
    return A.transpose()


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


def check_positive_definite(A) -> bool:
    """
    Boolean function for determining whether matrix A is positive definite.

    :param A: matrix A
    :return: true if yes, false otherwise
    """
    return (np.linalg.eigvals(A) > 0).all()


print("\nIs our matrix symmetric (A = A_T) ?", "Yes." if check_matrix_symmetry(M) else "No.")
print("Is our matrix positive definite?", "Yes." if check_positive_definite(M) else "No.")


def get_A_init(A):
    """

    :param A: initial matrix A, before it undergoes any changes
    :return: copy of matrix A
    """
    A_initial = np.copy(A)
    return A_initial


M_init = get_A_init(M)


def check_diagonal_is_positive(d) -> bool:
    """
    Boolean function determining whether a matrix A has exclusively positive elements on its main diagonal.

    :param d: diagonal of matrix
    :return: true if l(i,i) > 0 for all i, false otherwise
    """
    for elem in d:
        if elem == 0:
            return False
    return True


print("Is our matrix's diagonal positive? ", "Yes." if check_diagonal_is_positive(d) else "No.")


def library_cholesky_decomposition(A):
    """
    Function for getting the L, L_t Cholesky decomposition as computed by the numpy python library.

    :param A: matrix
    :return: L, L_t
    """
    L = scipy.linalg.cholesky(A, lower=True)     # lower triangular matrix in Cholensky decomposition for A:
    L_t = scipy.linalg.cholesky(A, lower=False)  # upper ~
    return L, L_t


# L, L_t = library_cholesky_decomposition(M)
# print("\nCholesky decomposition (using numpy): \n", L, '\n\n', L_t, '\n')


def cholesky_decomp(M):
    """
    Function for computing the Cholesky decomposition of a matrix A.

    :param A: matrix for which we perform the Cholesky decomposition
    :return: A as cholesky decomposition (lower and upper triangular matrix)
    """
    A = np.squeeze(np.asarray(M))
    # computing the lower triangular matrix dirrectly on matrix A:
    for i in range(0, n):
        if abs(A[i, i]) > eps:
            # computing elements on diagonal l(p,p):
            A[i, i] = math.sqrt(A[i, i] - np.dot(A[i, 0:i], A[i, 0:i], out=None))
            for j in range(i + 1, n):
                # l(i,p):
                A[j, i] = (A[j, i] - np.dot(A[j, 0:i], A[i, 0:i], out=None)) / A[i, i]
        else:
            print("Error! Division by 0, Cholesky decomposition not possible.")
    # populating matrix with 0
    for i in range(1, n):
        A[0:i, i] = 0.0
    M = np.asmatrix(A)
    return M


M = cholesky_decomp(M)
print("\nOur Cholesky decomposition (L, L_t): \n", M, "\n\n", get_transpose(M))


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


print("\ndet(M) = det(L) * det(L_t) =", compute_det_A(M, get_transpose(M)))


"""
L * y   = b  --> first
L_t * x = y  --> second
"""


def solve_Ly_equals_b(A, b):
    """
    Solve L * y = b  =>  find y*

    :param b: initial vector b
    :return: vector y*
    """
    Y = []
    L = np.squeeze(np.asarray(A))
    possible = True
    for row in range(0, n):
        y = b[row]
        for col in range(0, row):
            y -= np.dot(L[row, col], Y[col])
        if abs(L[row, row]) < eps:  # eps -> 0
            possible = False
            break
        else:  # if abs(L[row, row]) > eps: can perform division
            Y.append(y / L[row, row])
    if possible is True:
        return Y

# def solve_Lt_x_equals_y_star(L, Y):
#     """
#
#     :param L: lower triangular matrix (we will use its transpose -> upper triangular matrix)
#     :param Y: y*
#     :return: x* (solution)
#     """
#     L_t = get_transpose(L)  # upper triangular matrix
#     X = []
#     for row in range(n-1, -1, -1):
#         x = Y[row]
#         # print("Y[row] =", Y[row])
#         for col in range(0, row):
#             x -= np.dot(L_t[row, col], X[col])
#         if abs(L_t[row, row]) < eps:  # eps -> 0
#             break
#         else:
#             X.append(x / L_t[row, row])
#         # X.append(Y[row] - np.dot(L_t[row+1:n, row Y[row+1, n]) / L_t[row, row])
#     # X = np.array(X)
#     # X = np.asmatrix(X)
#     return X


def solve_Lt_x_equals_y_star(L, Y):  # works; checked
    """
    Solve L_t * x = Y -> find x


    :param L: lower triangular matrix (we will use its transpose -> upper triangular matrix)
    :param Y: y*
    :return: x* (solution)
    """
    L_t = get_transpose(L)  # upper triangular matrix
    # print(L_t)
    X = np.zeros(n)
    for row in range(0, n):
        x = Y
        # print("Y[row] =", Y[row])
        for col in range(0, row):
            x -= np.dot(L_t[row, col], X[col])
        if abs(L_t[row, row]) <= eps:
            break
        else:
            X[row] = x[row] / L_t[row, row]
    return X


def solve_system(B, b):
    # A will be
    A = np.copy(B)
    print("\nb =", b)
    Y = solve_Ly_equals_b(A, b)
    print("Y =", Y)
    A = np.copy(B)
    X = solve_Lt_x_equals_y_star(A, Y)  # function works with A_t
    print("X =", X)
    return X


print("\nM-init is now: \n", M_init)  # M_init
print("M is now: \n", M)            # lower triangular matrix

M_init2 = M_init

X_cholesky = solve_system(cholesky_decomp(M_init2), b)
# M is now the lower triangular matrix of M_init
print("\nX_cholesky is: ", X_cholesky)


def check_cholesky_is_correct(A, x_chol, b) -> bool:
    """
    Boolean function for checking whether our Cholesky decomposition is correct.
    A correct Cholesky decomposition of A = L * L_t will hold that A_init * A_Chol = b

    :param A: matrix A
    :return: true if matrix multiplication is same as b, false otherwise
    """
    product = np.matmul(A, x_chol)
    res = product.flatten()
    return bool((res == b).all())

# print("\nIs our Cholesky decomposition correct? ", "Yes." if check_cholesky_is_correct(M_init, ) else "No.")


print("Is Cholesky correct? ", "Yes" if check_cholesky_is_correct(M_init, X_cholesky, b) else "No")


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
    for index in range(0, n):  # column
        # getting vector b as a vector with 0s and a 1 on the position 'index'
        b = np.zeros(n)
        b[index] = 1.0
        # for each column in matrix A, we solve the system and the solution X will be col_i of A_inverse
        col_i = solve_system(A, b)  # equivalent to X_chol
        for row in range(0, n):
            A_inverse[row, index] = col_i[row]
    return A_inverse


print("\n Matrix M's inverse matrix (ours): \n", get_inverse_by_hand(M_init2))


def get_inverse_by_lib(A):
    """
    Function that returns the inverse matrix of matrix A as computed by python's numpy library.

    :param A: matrix A
    :return: inverse of A
    """
    return np.linalg.inv(A)


print("\n Matrix M's inverse matrix (numpy): \n", get_inverse_by_lib(M_init2))


def compute_norm_for_LLt_decomposition(A_init, b):
    """
    Function for computing the Euclidean norm ||A_init * x_cholesky - b||2 with the Python library
    (used for checking rate of success for our computation).
    For reference we will use epsilon = 10 ** (-9)

    :return: euclidean norm value
    """
    X_chol = solve_system(A_init, b)
    # X_chol = np.linalg.solve(A_init, b)   # computing x_cholesky
    mul = np.dot(A_init, X_chol)            # A_init * X_cholesky
    mul = np.subtract(mul, b)               # subtracting b
    val = np.linalg.norm(mul)               # computing euclidean norm on new result
    if val < pow(10, -9):
        print("\nNorm ||A_init * x_cholesky - b|| is ", val, ". \nComputation is correct with the error epsilon.")
    else:
        print("\nNorm ||A_init * x_cholesky - b|| is ",  val,  ". \nComputation is faulty.")


compute_norm_for_LLt_decomposition(M_init2, b)


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
        n = 3  # random.randint(3, 201)       # matrix size in [3, 200]
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
        compute_norm_for_approximation_of_cholesky_inverse(X_cholesky)

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
        n = int(input())
        print("m = ", end='')
        m = input()
        eps = 10 ** (m * (-1))
        b = []
        bv = str(input("b = "))
        bv.split(" ")
        for i in b:
            b.append(float(i))
        M = np.matrix([n, n])
        for row in range(0, n):
            for col in range(0, n):
                M[row, col] = float(input())
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

