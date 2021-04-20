import math
import random
from sympy import Symbol, Derivative


# def first_derivative(index, a):
#     """
#     Method for computing the numerical value of f'(a).
#
#     :param index: index of function f
#     :param a: point a
#     :return: f'(a)
#     """
#     value = a
#
#     # calculating the sympy derivative:
#     a = Symbol('a')
#
#     # getting the right function f:
#     if index == 1:
#         f = 1 / 3 * (a ** 3) - 2 * (a ** 2) + 2 * a + 3
#     elif index == 2:
#         f = a * a + math.sin(a)
#     else:
#         f = a ** 4 - 6 * (a ** 3) + 13 * a * a - 12 * a + 4
#     deriv = Derivative(f, a)
#     # returning the value of the derivative in point a:
#     return deriv.doit().subs({a: value})
#
#
# def second_derivative(index, a):
#     """
#     Method for computing the numerical value of f"(a)
#
#     :param index:
#     :param a:
#     :return: f"(a)
#     """
#     value = a
#
#     # calculating the sympy derivative:
#     a = Symbol('a')
#
#     # getting the right function f:
#     if index == 1:
#         f = 1 / 3 * (a ** 3) - 2 * (a ** 2) + 2 * a + 3
#     elif index == 2:
#         f = a * a + math.sin(a)
#     else:
#         f = a ** 4 - 6 * (a ** 3) + 13 * a * a - 12 * a + 4
#
#     deriv = Derivative(f, a)
#     deriv = Derivative(deriv, a)
#     # returning the value of the second derivative in point a:
#     return deriv.doit().subs({a: value})


def G(F, x, i):
    """
    Approximation formulae for the 1st derivative of F' employing its values.

    :param x: point x
    :param F: function F
    :param i:
    :return: F'(x)
    """
    h = pow(10, -6)
    if i == 1:
        return (3 * F(x) - 4 * F(x - h) + F(x - 2 * h)) / (2 * h)
    return (F(x + 2 * h) * (-1) + 8 * F(x + h) - 8 * F(x - h) + F(x - 2 * h)) / (12 * h)


# f = lambda x: 3 * (2 ** x) - 14 * x * x + math.sin(x)
# print(G(f, 3, 1), G(f, 3, 2), sep='\n')


def second_derivative(F, x):
    """
    Approximation formulae for the 2nd derivative of F"(x).

    :param F: function F
    :param x: parameter x of F(x)
    :return:
    """
    h = pow(10, -6)
    return (-F(x + 2 * h) + 16 * F(x + h) - 30 * F(x) + 16 * F(x - h) - F(x - 2 * h)) / (12 * pow(h, 2))


def dehghan_hajarian(F, epsilon):
    """

    :param F: function F
    :param epsilon: precision
    :return:
    """
    kmax = 10000

    # randomly choosing x0 s.t. it is in the neighbourhood of a root or 0 if we can't find one
    x0 = 0
    while G(F, x0, i=2) >= 0 and kmax >= 0:
        x0 = random.randint(-20, 20)
        kmax -= 1
    if G(F, x0, i=2) >= 0:
        print("Couldn't find x0 s.t. G(x0) < 0. ")
        x0 = 0

    # Dehgan-Hajarian algorithm:
    x = x0

    if abs(G(F, x, i=2)) < epsilon:
        return False
    # z:
    z = x + pow(G(F, x, i=2), 2) / \
        (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
    # Δx(k):
    delta_x_k = G(F, x, i=2) * (G(F, z, i=2) - G(F, x, i=2)) \
              / (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
    # x(k+1):
    x = x - delta_x_k

    print("initial x: ", x)
    kmax = 10000
    running = True
    while kmax > 0 and running:
        if epsilon <= abs(delta_x_k) < 10 ** 8 and kmax >= 0:
            if abs(G(F, x + G(F, x, i=2), i=2) - G(F, x, i=2)) < epsilon:
                return x
            z = x + pow(G(F, x, i=2), 2) / \
                (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
            delta_x_k = G(F, x, i=2) * (G(F, z, i=2) - G(F, x, i=2)) \
                        / (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
            x -= delta_x_k
            print("x =", x)
            print("delta =", delta_x_k)
            # next while-loop iteration:
            kmax -= 1
        else:
            break

    if delta_x_k < epsilon:
        return x  # x*
    return "divergence"


if __name__ == "__main__":
    # x = Symbol('x')
    for i in range(0, 3):
        if i == 0:
            f = lambda x : pow(x, 3) / 3 - 2 * pow(x, 2) + 2 * x + 3
        elif i == 1:
            f = lambda x: pow(x, 2) + math.sin(x)
        else:
            f = lambda x: pow(x, 4) - 6 * pow(x, 3) + 13 * pow(x, 2) - 12 * x + 4
        # print(f'Dehgan-Hajarian for f(x): {dehghan_hajarian(f, 10**(-6))}')
        x_star = dehghan_hajarian(f, 10**(-6))
        print("x* =", x_star) if type(x_star) == float else print("Divergence.")
        x_2nd_deriv = second_derivative(f, x_star)
        print("x* is a critical point of minimum\n", f'x" =', x_2nd_deriv) if x_2nd_deriv > 0 else \
            print("x* is not a point of minimum\n", f'x" =', x_2nd_deriv)

