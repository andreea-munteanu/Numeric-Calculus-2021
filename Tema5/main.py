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


def rotate(n, A, p, q, c, s):
    """
    Method for rotating matrix A (size n x n) until A[p][q] = A[q][p] = 0. Rotated matrix will be R.

    :param n: size of matrix A
    :param A: matrix A
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
    prod = np.matmul(R, A, R.transpose())
    return prod


def compute_U(U, R_T):
    """
    Method for computing U(k+1) = U x R_T(p, q)

    :param U: unitary matrix
    :param R_T: transpose of rotation matrix R(p, q)
    :return: U
    """
    prod = np.matmul(U, R_T)
    return prod


def jacobi_eigenvalues(n, A):
    """
    Method for obtaining the eigenvalues of symmetric matrix A using the jacobi method.

    :param n: number of columns, rows
    :param A: symmetric matrix A
    :return: eigenvalues for A as []
    """
    kmax = 10000
    U = np.identity(n)  # identity matrix

    # computing indices p and q (indices of the greatest non-diagonal element in absolute value):
    max = float('-inf')
    p, q = 0, 0
    for i in range(1, n):
        for j in range(0, i):
            if A[i][j] > max:
                max, p, q = A[i][j], i, j

    # computing angle theta (c = cos, s = sin, t = tan):
    alpha = (A[p][p] - A[q][q]) / (2 * A[p][q])
    t = (-1) * alpha + math.sqrt(alpha ** 2 - 1) if alpha >= 0 \
        else alpha * (-1) - math.sqrt(alpha ** 2 - 1)
    c = 1 / math.sqrt(1 + t)
    s = t / math.sqrt(1 + t)

    # while loop:
    while not check_diagonal_matrix(n, A) and kmax > 0:
        R = rotate(n, A, p, q, c, s)
        A = compute_A(A, R)
        U = compute_U(U, R.transpose())
        
        # compute p, q:
        max = float('-inf')
        p, q = 0, 0
        for i in range(1, n):
            for j in range(0, i):
                if A[i][j] > max:
                    max, p, q = A[i][j], i, j
        # computing c, s, t:
        alpha = (A[p][p] - A[q][q]) / (2 * A[p][q])
        t = (-1) * alpha + math.sqrt(alpha ** 2 - 1) if alpha >= 0 \
            else alpha * (-1) - math.sqrt(alpha ** 2 - 1)
        c = 1 / math.sqrt(1 + t)
        s = t / math.sqrt(1 + t)

        # next step in while loop:
        kmax -= 1


def assert_c_s(c, s):
    """
    :param c: cos(theta)
    :param s: sin(theta)
    :return: true if (c ** 2 + s ** 2 = 1), false otherwise
    """
    return c ** 2 + s ** 2 == 1


if __name__ == '__main__':
    p, n, eps, A = get_input()
    epsilon = 10 ** (-eps)
    print("n =", n, "\np =", p, "\nepsilon =", epsilon)
    # eigenvalues and eigenvectors:
    if p == n:
        if check_symmetry(p, n, A):
            print(jacobi_eigenvalues(n, A))
    # SVD:
    elif p > n:
        pass
    print("A:\n", A)



