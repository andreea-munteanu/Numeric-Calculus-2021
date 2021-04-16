import math
import random


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
    assert x0 < xn, "Invalid input"
    x = [0 for _ in range(0, n+1)]
    x[0], x[n] = x0, xn
    for i in range(n - 1, 0, -1):
        x[i] = random.uniform(x[i+1] - 1, x[i+1])
    y = [0 for _ in range(0, n+1)]
    for i in range(0, n+1):
        y[i] = function(index, x[i])
    return n, a, b, x, y


def horner(P, n, v):
    """
    Horner's method for computing P(v).

    :param P:
    :param n:
    :param v:
    :return: P(v)
    """
    res = 0
    for i in range(0, n):
        res = res * v + P[i]
    return res


""" ########################################### LEAST SQUARES INTERPOLATION #########################################"""


def least_squares_interpolation(a, b, x, y):
    """
    a = x0 < x1 < ... < xn = b

    :param a:
    :param b:
    :param x:
    :param y:
    :return:
    """
    assert a == x[0], "Invalid input for a."
    assert b == x[n], "invalid input for b."
    


if __name__ == '__main__':
    for index in range(1, 4):
        print('\n')
        n, a, b, x, y = get_input(index)
        print("n =", n, "a =", a, "b =", b)
        print("x: ", x)
        print("y: ", y)
        # print(n, x0, xn, a, b)
