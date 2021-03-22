import numpy as np
import re


# find number of lines in input file 'a.txt'
file = open("a.txt", "r")
Counter = 0
# Reading from file
Content = file.read()
CoList = Content.split("\n")
for i in CoList:
    if i:
        Counter += 1


def read_A(file):
    """
    Implementation with list of lists.

    :param file: input file 'a.txt'
    :return: matrix A as list of lists
    """
    f = open(file, "r")
    n = int(f.readline())  # matrix size
    # print("n =", n)  --> 2021
    line = f.readline()
    A = [[] for _ in range(n)]
    count = -1
    for _ in range(Counter - 1):  # for line in a.txt:
        count += 1
        line_i = f.readline().split(', ')
        val = line_i[0]
        row, col = int(line_i[1]), int(line_i[2])
        # print(count, val, row, col)
        row_count = -1
        for i in A:  # 'i' is a list containing tuples (col, val)
            row_count += 1
            if row == row_count:
                found_col = False
                for tup in i:
                    # if we reached an already registered (row, col) position:
                    if tup[0] == col:
                        found_col = True
                        # we save new val, delete old (col, val) and add new tuple (col, new_val):
                        val += tup[1]
                        i.remove(tup)
                        i.append((col, float(val)))
                # if col doesn't exist on row, we add the tuple (col, val) on row 'i':
                if not found_col:
                    i.append((col, float(val)))
    f.close()
    return n, A



# print("n = ", n)
# count = 0
# for i in A:
#     print(count, ": ", i)
#     count += 1
