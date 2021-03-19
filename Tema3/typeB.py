# vectors for tridiagonal matrix B:
a = []
b = []
c = []


def read_B_from_file(file, a, b, c):
    """
    Method for extracting input for tridiagonal matrix B from input file.

    :param file: input file 'b.txt'
    :param a: main diagonal
    :param b: diagonal above main by p positions
    :param c: diagonal below main by q positions
    :return: a, b, c
    """
    f = open(file, "r")
    n = int(f.readline())                           # 2021
    p, q = int(f.readline()), int(f.readline())     # 1, 1
    empty_line = f.readline()
    counter = 3 * n - p - q
    for _ in range(0, n):
        val = f.readline()
        a.append(float(val))
    # print("a =", a)

    empty_line = f.readline()
    for _ in range(0, n - q):
        val = f.readline()
        b.append(float(val))
    # print("b =", b)

    empty_line = f.readline()
    for _ in range(0, n - p):
        val = f.readline()
        c.append(float(val))
    # print("c =", c)

    return p, q, a, b, c


p, q, a, b, c = read_B_from_file('b.txt', a, b, c)