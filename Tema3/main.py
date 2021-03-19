import numpy as np
from typeA import n, A
from typeB import p, q, a, b, c

eps = 10 ** (-9)


def A_plus_B(n, A, p, q, a, b, c):
    """
    Method for computing sum A + B of sparse matrices.

    :param n: size of square matrices A and B
    :param A: matrix A stored as list of lists
    :param p: distance between main diagonal a and second diagonal b in matrix B
    :param q: distance between main diagonal a and third diagonal b in matrix B
    :param a: main diagonal in B (stored as list)
    :param b: second diagonal in B (stored as list)
    :param c: third diagonal in B (stored as list)
    :return: matrix (A + B)
    """
    SUM = np.copy.deepcopy(A)  # initially, SUM = A
    col = 0   # current col in for
    row = -1  # current row in for
    for i in SUM:
        row += 1
        col = 0
        for tup in i:
            # if col exists in SUM:
            if tup[0] == col:
                # main diagonal:
                if row == col:
                    val = tup[1] + a[col]
                    SUM.remove(tup)
                    SUM.append((col, val))
                # second diagonal (vector b):
                elif row == col - q:
                    val = tup[1] + b[col]
                    SUM.remove(tup)
                    SUM.append((col, val))
                # third diagonal (vector c):
                elif col == row - p:
                    val = tup[1] + c[col]
                    SUM.remove(tup)
                    SUM.append((col, val))
            # if col doesn't exist in SUM:
            else:
                # main diagonal:
                if row == col:
                    pass
                # second diagonal (vector b):
                elif row == col - q:
                    pass
                # third diagonal (vector c):
                elif col == row - p:
                    pass
    return SUM


def check_A_plus_B(file, sum_matrix) -> bool:
    """
    Boolean method for checking whether our computed sum is the same as the one in 'aplusb.txt'

    :param file: input file for A+B
    :param sum_matrix: A+B computed by us
    :return: true
    """
    C = A_plus_B(n, A, p, q, a, b, c)
    return True


def A_times_B(n, A, p, q, a, b, c):
    """
    Method for computing product A * B of sparse matrices.

    :param n: size of square matrices A and B
    :param A: matrix A stored as list of lists
    :param p: distance between main diagonal a and second diagonal b in matrix B
    :param q: distance between main diagonal a and third diagonal b in matrix B
    :param a: main diagonal in B (stored as list)
    :param b: second diagonal in B (stored as list)
    :param c: third diagonal in B (stored as list)
    :return: matrix A*B
    """
    P = np.copy.deepcopy(A)
    pass


def check_A_times_B(file, prod_matrix):
    """
    Boolean method for checking whether our computed matrices product is the same as the one in 'aorib.txt'

    :param file: input file for A+B
    :param sum_matrix: A+B computed by us
    :return: true
    """
    C = A_plus_B(n, A, p, q, a, b, c)
    return True


