import numpy as np
from typeA import read_A
from typeB import read_B_from_file
import filecmp

# error epsilon
eps = 10 ** (-9)
# vectors for tridiagonal matrix B:
a = []
b = []
c = []


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
            if val != 0:
                row.append((index, val))

    # adding elements on diagonal b: (successful)
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] < n - q and tup[0] == index + q:
                found_tup = tup
                found_index = True
        if not found_index:
            if index < n - 1:
                row.append((index + q, b[index]))
        else:
            if index < n:
                val = found_tup[1] + b[index]
                row.remove(found_tup)
                if val != 0:
                    row.append((index + q, val))

    # adding elements on diagonal c: (successful)
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] > p - 1 and tup[0] == index - p:
                found_tup = tup
                found_index = True
        if not found_index:
            if index > 0:
                row.append((index - p, c[index - p]))
        else:
            if index < n:
                val = found_tup[1] + c[index - p]
                row.remove(found_tup)
                if val != 0:
                    row.append((index - p, val))

    # sorting elements on row by col
    for i in SUM:
        i.sort(key=lambda tup: tup[0])
    return SUM


def turn_B_into_list_of_lists(n, p, q, a, b, c):
    """
    Method for turning vectors a[n], b[n-1], c[n-1] representing B into a list of list similar to A.

    :param n: size of matrices
    :param p:
    :param q:
    :param a:
    :param b:
    :param c:
    :return:
    """
    B = []
    for index in range(0, n):
        row = []
        # adding element on main diagonal
        row.append((index, a[index]))
        # adding element on diagonal b
        if index < n - q:
            row.append((index + q, b[index]))
        # adding element on diagonal c
        if p - 1 < index < n:  # p = 1
            row.append((index - p, c[index - p]))
        B.append(row)
    for i in B:
        i.sort(key=lambda tup: tup[0])
    return B


def A_times_B(n, A, p, q, a, b, c):
    """
    Method for computing product A * B of sparse matrices.
    P(i, j) = sum( A(i, k) * B(k, i) ), k = 1..n

    :param n: size of square matrices A and B
    :param A: matrix A stored as list of lists
    :param p: distance between main diagonal a and second diagonal b in matrix B
    :param q: distance between main diagonal a and third diagonal b in matrix B
    :param a: main diagonal in B (stored as list)
    :param b: second diagonal in B (stored as list)
    :param c: third diagonal in B (stored as list)
    :return: matrix A*B
    """
    C = []
    B = turn_B_into_list_of_lists(n, p, q, a, b, c)
    for i, row in enumerate(A):
        for j, col in enumerate(B):
            if row[i][1] == col[]


        # s = 0  # element C(i, j)
        # for j, col in enumerate(B):
        #     found = None
        #     for tup1 in row:
        #         if tup1[0] == j:
        #             found = tup1
        #             break
        #     if found is None:
        #         pass
        #     else:
        #         if j == i:
        #             s += found[1] * col[1]








    # P = []
    # row_index = 0  # row count
    # for i in A:
    #     P_row = []
    #     for index in range(len(i)):
    #         val = 0
    #         for tup in i:
    #             if tup[0] == index:             # col == row --> vector a
    #                 val += tup[1] * a[index]
    #             elif tup[0] == index + q:       # col == row + p --> vector b
    #                 val += tup[1] * b[index]
    #             elif tup[0] == index - p:
    #                 val += tup[1] * c[index]    # col == row - q --> vector b
    #         P_row.append((row_index, val))
    #     P.append(P_row)
    #     row_index += 1
    # return P


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
                print(tup[1], ", ", row, ", ", tup[0], file=f, sep='')


def compare_files(file1, file2) -> bool:  # works
    """
    Boolean method for checking whether two files are identical

    :param file1: text file #1
    :param file2: text file #2
    :return: true if our output is the same as the one in the given file, false otherwise
    """
    same = True
    with open(file1) as f1, open(file2) as f2:
        count = 0
        for x, y in zip(f1, f2):
            x = x.strip()
            y = y.strip()
            count += 1
            if count > 2:
                line_x = x.split(', ')
                val1 = float(line_x[0])
                row1, col1 = int(line_x[1]), int(line_x[2])
                line_y = y.split(', ')
                val2 = float(line_y[0])
                row2, col2 = int(line_y[1]), int(line_y[2])
                if abs(val1 - val2) >= eps or row1 != row2 or col1 != col2:
                    same = False
                    break
    return same


if __name__ == "__main__":
    # read n, A, p, q, a, b, c from text files:
    n, A = read_A('a.txt')
    p, q, a, b, c = read_B_from_file('b.txt', a, b, c)
    # compute sum:
    SUM = A_plus_B(n, A, p, q, a, b, c)
    # export sum to file:
    export_matrix_to_file('computed_sum.txt', n, SUM)
    # compare our result to the one in the given file:
    print("Is the sum correct? ", "Yes" if compare_files('computed_sum.txt', 'aplusb.txt') else "No")
    PROD = A_times_B(n, A, p, q, a, b, c)
    export_matrix_to_file('computed_product.txt', n, PROD)
    print("Is product correct?", "Yes" if compare_files('computed_product.txt', 'aorib.txt') else "No")


