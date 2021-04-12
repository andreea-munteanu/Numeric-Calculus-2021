import math
import random


def G(F, x, i=2):
    """
    Approximation formulae for the 1st derivative of F' employing its values.

    :param x:
    :param F:
    :param i:
    :return:
    """
    h = pow(10, -6)
    if i == 1:
        return (3 * F(x) - 4 * F(x - h) + F(x - 2 * h)) / (2 * h)
    return (F(x + 2 * h) * (-1) + 8 * F(x + h) - 8 * F(x - h) + F(x - 2 * h)) / (12 * h)


def second_derivative(F, x):
    """
    Approximation formulae for the 2nd derivative of F":
    :param F: function F
    :param x: parameter x of F(x)
    :return:
    """
    h = pow(10, -6)
    return (-F(x + 2 * h) + 16 * F(x + h) - 30 * F(x) + 16 * F(x - h) - F(x - 2 * h)) / (12 * pow(h, 2))


def dehghan_hajarian(F, epsilon):
    """

    :param F:
    :param epsilon: precision
    :return:
    """
    kmax = 10000

    # randomly choosing x0 s.t. it is in the neighbourhood of a root or 0 if we can't find one
    x0 = 0
    while G(F, x0) >= 0 and kmax > 0:
        x0 = random.randint(-20, 20)
        kmax -= 1
    if G(F, x0) >= 0:
        print("Couldn't find x0 s.t. G(x0) < 0. ")
        x0 = 0

    # Dehgan-Hajarian algorithm:
    x = x0

    if abs(G(F, x)) < epsilon:
        return False
    # z:
    z = x + pow(G(F, x, i=2), 2) / \
        (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
    # Δx(k):
    delta_x_k = G(F, x, i=2) * (G(F, z, i=2) - G(F, x, i=2)) \
              / (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
    # x(k+1):
    x = x - delta_x_k

    kmax = 10000
    running = True
    while kmax > 0 and running:
        if epsilon <= abs(delta_x_k) < 10 ** 8:
            if abs(G(F, x + G(F, x)) - G(F, x)) < epsilon:
                return x
            z = x + pow(G(F, x, i=2), 2) / \
                (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
            delta_x_k = G(F, x, i=2) * (G(F, z, i=2) - G(F, x, i=2)) \
                        / (G(x + G(F, x, i=2), x, i=2) - G(F, x, i=2))
            x = x - delta_x_k
            # next while-loop iteration:
            kmax -= 1
        else:
            break

    if delta_x_k < epsilon:
        return x
    return True


if __name__ == "__main__":
    f = lambda x: pow(x, 3) / 3 - 2 * pow(x, 2) + 2 * x + 3
    # f = lambda x: pow(x, 2) + math.sin(x)
    # f = lambda x: pow(x, 4) - 6 * pow(x, 3) + 13 * pow(x, 2) - 12 * x + 4
    print(f'Dehgan-Hajarian for f(1) = {f(1)}: {dehghan_hajarian(f, 10**-6)}')

    # def switch(function):
    #     switcher = {
    #         1: lambda x: pow(x, 3) / 3 - 2 * pow(x, 2) + 2 * x + 3,
    #         2: lambda x: pow(x, 2) + math.sin(x),
    #         3: lambda x: pow(x, 4) - 6 * pow(x, 3) + 13 * pow(x, 2) - 12 * x + 4
    #     }
    #     print(switcher.get(function, "Invalid function number."))
