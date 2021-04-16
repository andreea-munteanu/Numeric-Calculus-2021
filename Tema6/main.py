import math


def get_input(index):
    """
    Method for extracting input from text file 'input{index}.txt'

    :param index: number of input file.
    :return: n, x0, xn, a, b
    """
    file = open(f'input{index}.txt', "r")
    n = file.readline()
    n = int(n)
    x0, a = file.readline().split()
    x0, a = float(x0), float(a)
    xn, b = file.readline().split()
    xn, b = float(xn), float(b)
    return n, x0, xn, a, b


def function(i, x):
    """
    Method for calculating the value of a function with the index 'i' in point x.

    :param i: example number
    :param x: value of x for which we compute the function
    :return: f(x)
    """
    assert i in [1, 2, 3], "Invalid index."
    if i == 1:
        return x ** 2 - 12 * x + 30
    elif i == 2:
        return math.sin(x) - math.cos(x)
    else:
        return 2 * (x ** 3) - 3 * x + 15


def horner(P, n, v):
    """
    Horner's method for computing P(v).

    :param P: polynomial
    :param n: rank of P
    :param v: the point
    :return: P(v)
    """
    result = P[0]
    for i in range(1, n):
        result = result * v + P[i]
    return result


if __name__ == '__main__':
    for index in range(1, 4):
        n, x0, xn, a, b = get_input(index)
        # print(n, x0, xn, a, b)
