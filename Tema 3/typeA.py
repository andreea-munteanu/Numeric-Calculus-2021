import numpy as np
import re


def read_A_from_file(file):
    """
    Implementation with nested dictionaries.

    :param file: input file
    :return: matrix A as nested dictionary
    """
    f = open(file, "r")
    n = int(f.readline())  # matrix size
    # print("n =", n)  --> 2021
    space = f.readline()
    line = f.readline()
    A = {}
    count = 0
    while line:
        line_i = f.readline().split(', ')
        val = float(line_i[0])
        row, col = int(line_i[1]), int(line_i[2])
        print(count, val, row, col)
        if row in A.keys():
            if col in A.keys():
                pass
        # else:
        #     A[row] = {(val, col)}
        #print(A)
        # else:
        #     new_row = dict()
        #     new_row[col] = val
        #     A[row] = new_row
        count += 1
    # print("A: \n", A)
    f.close()
    return n, A


def read_A(file):
    """
    Implementation with list of lists.

    :param file: input file 
    :return: matrix A as list of lists
    """
    f = open(file, "r")
    n = int(f.readline())  # matrix size
    # print("n =", n)  --> 2021
    line = f.readline()
    A = [[]] * n
    count = 0
    while line:
        line_i = f.readline().split(', ')
        val = float(line_i[0])
        row, col = int(line_i[1]), int(line_i[2])
        # print(count, val, row, col)
        row_count = -1
        for i in A:  # 'i' is a list containing tuples (col, val)
            row_count += 1
            if row == row_count:
                found_col = False
                for tup in i:
                    # if we reached an already registered (row, col) position, we add 'val':
                    if tup[0] == col:
                        # we save new val, delete old (col, val) and add new tuple (col, new_val):
                        val += tup[1]
                        i.remove(tup)
                        i.append((col, val))
                # if col doesn't exist on row, we add the tuple (col, val) on row 'i':
                if not found_col:
                    i.append((col, val))
    # check that every line has at least 20 non-zero values:
    count = -1
    for i in A:
        count += 1
        if len(i) < 20:
            print("Less that 20 non-zero elements on line", count)
    print(A)
    f.close()


read_A('a.txt')
# print(n, A)
