import random
import numpy as np


epsilon = 10 ** (-5)


def horner(x, coef):
    """
    Method for computing Horner.

    :param x: point x
    :param coef: polynomial (vector of coefficients)
    :return:
    """
    b_i = coef[0]
    for cf in range(1, len(coef)):
        b_i = coef[cf] + b_i * x
    return b_i


def generate_polynomial(coef):
    """
    Method for o
    :param coef:
    :return:
    """
    p = np.poly1d(coef)
    return p


def derivative(coef):
    """
    Method for computing the derivative of a certain polynomial in numpy.

    :param coef: polynomial P
    :return: P'
    """
    p = np.polyder(coef)
    return p


def olver(coef, P, P_deriv1_coef, P_deriv2_coef, iteration):
    """
    Method for implementing Olver's algorithm.

    :param coef: polynomial P (as vector of coefficients)
    :param P: value of polynomial
    :param P_deriv1_coef: P'
    :param P_deriv2_coef: P"
    :param iteration: number of iteration in course
    :return:
    """

    if iteration == 0:
        return "divergence"

    A = max([abs(c) for c in coef[1:len(coef)]])
    R = (abs(coef[0]) + A) / abs(coef[0])
    x = random.uniform(-R, R)
    kmax = 100
    delta_x = 0
    running = True

    while running:

        # computing polynomial P in point x and its first and second derivatives:
        P_val = horner(x, coef)
        P_first_derivative = horner(x, P_deriv1_coef)
        P_second_derivative = horner(x, P_deriv2_coef)

        # computing c_k, delta_x and x:
        c_k = (P_val ** 2) * P_second_derivative / (P_first_derivative ** 3)
        delta_x = P_val / P_first_derivative + 0.5 * c_k
        x = x - delta_x

        kmax -= 1

        if abs(P_first_derivative) <= epsilon:
            return olver(coef, P, P_deriv1_coef, P_deriv2_coef, iteration - 1)

        if not (epsilon <= abs(delta_x) <= 10 ** 8 and kmax >= 0):
            running = False

    if abs(delta_x) < epsilon:
        return x
    else:
        # print("divergence")
        return olver(coef, P, P_deriv1_coef, P_deriv2_coef, iteration - 1)


coef1 = [1.0, -6.0, 11.0, -6.0]
coef2 = [42.0 / 42, -55.0 / 42, -42.0 / 42, 49.0 / 42, -6.0 / 42]
coef3 = [8.0 / 8, -38.0 / 8, 49.0 / 8, -22.0 / 8, 3.0 / 8]
coef4 = [1.0, -6.0, 13.0, -12.0, 4.0]


if __name__ == "__main__":
    for coefficient in range(1, 5):
        print(f'############## Example {coefficient} ##############')
        pol = None
        if coefficient == 1:
            pol = generate_polynomial(coef1)
        elif coefficient == 2:
            pol = generate_polynomial(coef2)
        elif coefficient == 3:
            pol = generate_polynomial(coef3)
        elif coefficient == 4:
            pol = generate_polynomial(coef4)
        print("Polynomial: \n", pol)

        P_deriv1 = derivative(pol)
        P_deriv2 = derivative(P_deriv1)

        # computing derivative1 of each term in polynomial -> coefficients:
        P_deriv1_coef = []
        for i in P_deriv1.coef:
            P_deriv1_coef.append(i)

        # computing derivative2 of each term in polynomial -> coefficients:
        P_deriv2_coef = []
        for i in P_deriv2.coef:
            P_deriv2_coef.append(i)

        solutions = []
        for i in range(10000):
            solution = olver(coef4, pol, P_deriv1_coef, P_deriv2_coef, 1000)
            if solution == "divergence":
                print(solution)
            new = True
            for i in solutions:
                if abs(i - solution) <= epsilon:
                    new = False
            if new:
                solutions.append(solution)

        # writing in file:
        f = open("solutions.txt", "w")
        f.write(str([i for i in solutions]))
        f.close()

        print(solutions)
