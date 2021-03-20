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
    # initially, SUM = A
    SUM = A
    # adding elements on main diagonal (successful)
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] == index:
                found_tup = tup
                found_index = True
        if not found_index:
            row.append((index, a[index]))
        else:
            val = found_tup[1] + a[index]
            row.remove(found_tup)
            row.append((index, val))
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] > 0 and tup[0] == index + q:
                found_tup = tup
                found_index = True
        if not found_index:
            if index < n - 1:
                row.append((index + q, b[index]))
        else:
            if index < n - 1:
                val = found_tup[1] + b[index]
                row.remove(found_tup)
                row.append((index + q, val))

    # adding elements on diagonal c: (successful)
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] == index - p:
                found_tup = tup
                found_index = True
        if not found_index:
            if index > 0:
                row.append((index - p, c[index - p]))
        else:
            if index < n - 1:
                val = found_tup[1] + c[index - p]
                row.remove(found_tup)
                row.append((index - p, val))

    # sorting elements on row by col
    for i in SUM:
        i.sort(key=lambda tup: tup[0])
    return SUM


def check_A_plus_B(file, SUM) -> bool:
    """
    Boolean method for checking whether our computed sum is the same as the one in 'aplusb.txt'

    :param file: input file for A+B
    :param SUM: A+B computed by us
    :return: true
    """
    pass


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


