import math
import random
import numpy as np
from sympy import Symbol, Derivative


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

    # extracting file input:
    file = open(f'input{index}.txt', "r")
    n = file.readline()
    n = int(n)
    x0, a = file.readline().split()
    x0, a = float(x0), float(a)
    xn, b = file.readline().split()
    xn, b = float(xn), float(b)

    # randomly generating x1, ..., x(n-1) s.t. x(i) > x(i-1):
    x = [0 for _ in range(0, n + 1)]
    x[0], x[n] = x0, xn
    for i in range(n - 1, 0, -1):
        x[i] = random.uniform(x[i + 1] - 1, x[i + 1])

    # computing y by y[i] = f(x[i]):
    y = [0 for _ in range(0, n + 1)]
    for i in range(0, n + 1):
        y[i] = function(index, x[i])

    # checking file input:
    assert x0 < xn, "Invalid input"
    assert a == x[0], "Invalid input for a."
    assert b == x[n], "invalid input for b."

    # return values:
    return n, a, b, x, y


def horner(P, n, v):
    """
    Horner's method for computing P(v).

    :param P: polynomial P
    :param n: size of P
    :param v: point in which we compute the polynomial
    :return: P(v)
    """
    res = 0
    for i in range(0, n):
        res = res * v + P[i]
    return res


""" ########################################### LEAST SQUARES INTERPOLATION #########################################"""


def least_squares_interpolation(x, y, m, n):
    """
    a = x0 < x1 < ... < xn = b
    ¯x ∈ [a, b]

    :param x: x = [x0, x1, ..., xn]
    :param y: y = f(x0), f(x1), ..., f(xn)
    :param m: size of Ba = f
    :param n: size of x and y
    :return:
    """
    # getting Ba:
    B = [[1 if i == 0 else 0
          for i in range(0, m + 1)] for _ in range(0, n + 1)]

    for i in range(0, n + 1):
        for j in range(1, m + 1):
            B[i][j] = x[i] ** j
    B = np.array(B)
    print(B)

    # getting right form of y:
    Y = [[y[i]] for i in range(0, n + 1)]
    Y = np.array(Y)

    B_T = np.transpose(B)
    aux1 = np.dot(B_T, B)
    aux2 = np.dot(B_T, Y)

    # solving the system:
    res = np.linalg.solve(aux1, aux2)
    res = ([i[0] for i in res])
    res.reverse()
    return res


""" ###################################### QUADRATIC SPLINE FUNCTIONS OF CLASS C1 ###################################"""


def first_derivative(index, a):  # works; checked
    """
    Method for computing the numerical value of f'(a) = d_a.

    :param index: index of function f
    :param a: point a
    :return: f'(a)
    """
    value = a

    # calculating the sympy derivative:
    a = Symbol('a')

    # getting the right function f:
    if index == 1:
        f = a ** 2 - 12 * a + 30
    elif index == 2:
        f = math.sin(a) - math.cos(a)
    else:
        f = 2 * (a ** 3) - 3 * a + 15

    deriv = Derivative(f, a)
    # returning the value of the derivative in point a:
    return deriv.doit().subs({a: value})
    # return derivative(function(index, a), a, dx=1e-6, n=1)  --> doesn't work god knows why


if __name__ == '__main__':
    for index in range(1, 4):
        print('\n')
        n, a, b, x, y = get_input(index)
        print("n =", n, "a =", a, "b =", b)
        print("x: ", x)
        print("y: ", y)
        Pm = least_squares_interpolation(x, y, 10, n)
        print("Pm: ", Pm)
        print("Pn(x) =", horner(Pm, n, 2))
        # print("|Pn(x) - f(x)| =", abs(horner(Pm, n, 2) - horner([1, -12, 30, 0, 12], n, 2)))
        # print("derivative: ", first_derivative(index, 10))
        # print(n, x0, xn, a, b)
