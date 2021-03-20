import numpy as np
from typeA import n, A
from typeB import p, q, a, b, c
import filecmp

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
            # only adding non-zero values in sparse matrix SUM:
            if val != 0:
                row.append((index, val))

    # adding elements on diagonal b: (successful)
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
                # only adding non-zero values in sparse matrix SUM:
                if val != 0:
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
                if val != 0:
                    row.append((index - p, val))

    # sorting elements on row by col
    for i in SUM:
        i.sort(key=lambda tup: tup[0])
    return SUM


def export_matrix_to_file(output_file, n, mat):
    """
    Method for printing sparse matrix mat[n][n] in file text output_file.

    :param output_file: output_file containing our computed matrix
    :param n: square matrix size
    :param mat: output matrix to be printed
    :return:
    """
    with open(output_file, 'w') as f:
        print(n, file=f)
        print("", file=f)
        row = -1
        for i in mat:
            row += 1
            for tup in i:
                print(tup[1] + ", " + row, end='', file=f)
                print(", ", end='', file=f)
                print(tup[0], file=f)


def check_with_file(file1, file2) -> bool:
    """
    Boolean method for checking whether two files are identical

    :param file1: text file #1
    :param file2: text file #2
    :return: true if our output is the same as the one in the given file, false otherwise
    """
    return filecmp.cmp(file1, file2)  # NU E BUN, TREBUIE FOLOSIT EPSILON


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


if __name__ == "__main__":
    # compute sum:
    SUM = A_plus_B(n, A, p, q, a, b, c)
    # export sum to file:
    export_matrix_to_file('computed_sum.txt', n, SUM)
    # check that sum is the same as the
