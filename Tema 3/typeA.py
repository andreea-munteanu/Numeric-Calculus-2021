import numpy as np
import re


def read_A_from_file(file):
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
    f = open('a.txt', 'r').readlines()
    lines = len(f)


read_A_from_file('a.txt')
# print(n, A)