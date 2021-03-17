import numpy as np

# vectors for tridiagonal matrix B:
a = []
b = []
c = []


def read_B_from_file(file):
    f = open(file, "r")
    n = int(f.readline())                           # 2021
    p, q = int(f.readline()), int(f.readline())     # 1, 1



read_B_from_file('b.txt')
