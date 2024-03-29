import copy
import math
import numpy as np


def get_input():
    """
    Method for extracting input from text file.

    :return: size p x n of matrix A, computation error eps, matrix A
    """
    file = open("input.txt", "r")
    # A[p][n], computation error epsilon:
    p, n, eps = file.readline().split()
    n, p, eps = int(n), int(p), int(eps)
    A = []
    for i in range(p):
        line = file.readline().split()
        row = []
        for elem in line:
            row.append(float(elem))
        A.append(row)
    A = np.array(A)
    return p, n, eps, A


def check_symmetry(p, n, A):
    """
    Method for asserting that matrix A is symmetric (must also be square).

    :return: true if A is symmetric, false otherwise
    """
    A_T = A.transpose()
    if p == n:
        return (A == A_T).all()
    return False


def check_diagonal_matrix(n, A):
    """
    Method for asserting whether square matrix A is a diagonal matrix (zero values on all non-main-diagonal positions).

    :param n: size of square matrix A
    :param A: matrix A
    :return: true if A is a diagonal matrix, false otherwise
    """
    for i in range(0, n):
        for j in range(0, n):
            if i != j and A[i][j] != 0:
                return False
    return True


def get_A_init(A):
    """
    Method for getting A_init (copy of initial A).

    :param A: matrix A
    :return: A_init
    """
    A_init = copy.deepcopy(A)
    return A_init


def rotate(n, p, q, c, s):
    """
    Method for rotating matrix A (size n x n) until A[p][q] = A[q][p] = 0. Rotated matrix will be R.

    :param n: size of matrix A
    :param p: row index
    :param q: col index
    :param c: cos(theta)
    :param s: sin(theta)
    :return: R
    """

    # A(0) = A
    # A(k+1) = R(p, q, theta) x A(k) * R_T(p, q, theta)

    R = [[0 for _ in range(0, n)]] * n
    for i in range(0, n):
        for j in range(0, n):
            if i == j and i != p and i != q:
                R[i][j] = 1
            elif i == j and i in [p, q]:
                R[i][j] = c
            elif i == p and j == q:
                R[i][j] = s
            elif j == p and i == q:
                R[i][j] = -s
            else:
                R[i][j] = 0
    R = np.array(R)
    return R


def compute_A(A, R):
    """
    Method for computing A(k+1) = R(p,q) x A(k) x R_T(p, q)

    :param A: symmetric matrix A
    :param R: rotation matrix R(p, q)
    :return: A(k+1)
    """
    prod = np.matmul(R, A)
    prod_fin = np.matmul(prod, R.transpose())
    return prod_fin


def compute_U(U, R_T):
    """
    Method for computing U(k+1) = U x R_T(p, q)

    :param U: unitary matrix
    :param R_T: transpose of rotation matrix R(p, q)
    :return: U
    """
    prod = np.matmul(U, R_T)
    return prod


def compute_A_final(U, A_init):
    """
    Method for computing A(final) = U_T x A_init x U.

    :param U: unitary matrix
    :param A_init: initial matrix A
    :return: A(final)
    """
    prod = np.matmul(U.transpose(), A_init)
    prod_fin = np.matmul(prod, U)
    return prod_fin


def assert_c_s(c, s, epsilon):
    """
    :param c: cos(theta)
    :param s: sin(theta)
    :return: true if (c ** 2 + s ** 2 = 1), false otherwise
    """
    return c ** 2 + s ** 2 in (1 - epsilon, 1 + epsilon)


def jacobi(n, A):
    """
    Method for obtaining the eigenvalues and eigenvectors of symmetric matrix A using the jacobi method.

    :param n: number of columns, rows
    :param A: symmetric matrix A
    :return: eigenvalues and eigenvectors for A
    """
    kmax = 10000            # maximum number of while-loop iterations
    U = np.identity(n)      # identity matrix
    A_init = get_A_init(A)  # A_init

    # computing indices p and q (indices of the greatest non-diagonal element in absolute value):
    max = float('-inf')
    p, q = 0, 0
    for i in range(1, n):
        for j in range(0, i):
            if A[i][j] > max:
                max, p, q = A[i][j], i, j

    # computing angle theta (c = cos, s = sin, t = tan):
    alpha = (A[p][p] - A[q][q]) / (2 * A[p][q])
    t = (-1) * alpha + math.sqrt(alpha * alpha - 1) if alpha >= 1 \
        else alpha * (-1) - math.sqrt(alpha ** 2 - 1)
    c = 1 / math.sqrt(1 + t)
    s = t / math.sqrt(1 + t)

    while not check_diagonal_matrix(n, A) and kmax > 0:
        R = rotate(n, p, q, c, s)
        A = compute_A(A, R)
        U = compute_U(U, R.transpose())

        # compute p, q:
        max = float('-inf')
        p, q = 0, 0
        for i in range(1, n):
            for j in range(0, i):
                if A[i][j] > max:
                    max, p, q = A[i][j], i, j

        # A[p][q] = 0 case: A is a non-diagonal matrix -> algorithm stops
        if A[p][q] > epsilon:  # with this, we could lose the check_diagonal_matrix(n, A) test
            pass
        else:
            break

        # computing c, s, t:
        alpha = (A[p][p] - A[q][q]) / (2 * A[p][q])
        t = (-1) * alpha + math.sqrt(alpha ** 2 - 1) if alpha >= 1 \
            else alpha * (-1) - math.sqrt(alpha ** 2 - 1)
        c = 1 / math.sqrt(1 + t)
        s = t / math.sqrt(1 + t)

        # check that s**2 + c ** 2 = 1 (with computation error epsilon)
        if assert_c_s(c, s, epsilon):
            pass
        else:
            raise ValueError("c ** 2 + s** 2 should be 1.")

        # next step in while loop:
        kmax -= 1

    # compute A(final):
    A_final = compute_A_final(U, A_init)

    # A_final is an (approximately) diagonal matrix, the values from the diagonal are approximations of the eigenvalues
    # of matrix A, and the columns of matrix U (orthogonal matrix) are approximations of the corresponding eigenvectors.
    e_values = A_final.diagonal()
    e_vectors = []
    for col in range(0, n):
        e_vectors.append(U[:, col])

    return e_values, e_vectors


def SVD(p, n, A, b):
    """
    SVD for matrix A is:
    A = U x S x V_T,  with U[p][p], S[p][n], V[n][n]

    :param p: number of rows in A
    :param n: number of columns in A
    :param A: matrix A
    :param b: Ax = b
    :return:
    """
    assert p != n, "SVD cannot run."
    U, S, V_T = np.linalg.svd(A)
    print("\nU: ", U, sep='\n')
    print("\nS: ", S, sep='\n')
    print("\nV_T: ", V_T, sep='\n')

    # rank A is the number of strictly positive singular values (values on main diagonal of S):
    rank = 0
    min = float("inf")
    max = float("-inf")
    for elem in S:
        if elem > 0:
            rank += 1
            if elem < min:
                min = elem
        if elem > max:
            max = elem

    print("rank =", rank)

    # conditioning number: max/min
    conditioning_number = max/min
    print("conditioning number =", conditioning_number)

    # Moore-Penrose pseudo-inverse of matrix A:
    SS = np.zeros((p, n))
    for i in range(0, rank):
        SS[i][i] = 1 / S[i]
    PI_A = np.dot(np.dot(V_T.transpose(), SS), U.transpose())  # pseudo-inverse of A

    # system solution:
    sol = np.dot(PI_A, b)

    return rank, conditioning_number, PI_A, sol


if __name__ == '__main__':
    p, n, eps, A = get_input()
    epsilon = 10 ** (-eps)
    print("n =", n, "\np =", p, "\nepsilon =", epsilon)
    print("A:\n", A, end='\n')
    # eigenvalues and eigenvectors:
    if p == n:
        if check_symmetry(p, n, A):
            eigenvalues, eigenvectors = jacobi(n, A)
            print("eigenvalues: ", eigenvalues)
            print("eigenvectors: ", eigenvectors)
    # SVD:
    elif p > n:
        print("value for b (Ax = b): ", end='')
        b = np.array([1, 1, 1, 1])
        print(b)
        rank, conditioning_number, PI_A, sol = SVD(p, n, A, b)
        print("rank =", rank, "\nconditioning number =", conditioning_number,
              "\nPseudo-inverse of A:\n", PI_A, "\nsolution for Ax = b:\n", sol)




