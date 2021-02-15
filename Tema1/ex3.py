import math
import random

pi = math.pi  # 3.141592653589793
pos_inf = float('inf')  # +∞
neg_inf = float('-inf')  # -∞


def bring_in_interval(a):
    """
    :param a: number to be transformed into (-pi/2, pi/2)
    :return: number a in (-pi/2, pi/2);
             returns x itself, not tan(x)

    we know: tan(x + k*pi) = tan(x)
    """
    k = 0  # periodicity factor

    # pare sa mearga; sper
    if a > 0:
        while k * pi <= a:
            k += 1
        return k * pi - a

    # merge pe exemplul facut la lab, sper sa fie bun
    else:
        while k * pi >= a:
            k -= 1
        return k * pi - a


print(bring_in_interval((-5 * pi) / 3))


def generate_no_in_range():
    """
    :return: list of floats representing 10000 pseudo-randomly generated numbers in [-10000, 10000] and brought into (-pi/2, pi/2)
    """
    values = [random.randrange(-10000, 10001) for _ in range(1, 10001)]
    # values = random.sample(range(-7000, 7000), 10000)
    # print(values)
    for i in range(len(values)):
        # if x[i] in (-pi/2, pi/2) --> okay; otherwise bring into interval:
        if -pi / 2 < values[i] > pi / 2:
            values[i] = bring_in_interval(values[i])
        # separate cases for x = pi/2 and x = -pi/2:
        elif values[i] == pi / 2:
            values[i] = pos_inf
        elif values[i] == -pi / 2:
            values[i] = neg_inf
    return values


x = generate_no_in_range()
print(x)

"""
     ________________________________________ THE CONTINUED FRACTIONS METHOD __________________________________________

"""

epsilon = pow(10, -13)  # set precision as 10^(-13)


def calculate_tangent_lorentz(x, epsilon):
    """
    b[0] = x/(1+)
    a[1], a[2], ..., a[n] = -x^2
    b[1] = 1+
    b[2] = 3+ etc.
    """
    j = 0
    a = x[1] * (-x[1])
    b = x[j]
    f = b
    C = b
    D = 0
    mic = pow(10, -12)
    if f == 0:
        f = mic
    C = f
    D = 0
    j = 1

    # simulating a do-while:

    D = b + a * D
    if D == 0:
        D = mic
    C = b + (a / C)
    if C == 0:
        C = mic
    D = 1 / D
    delta_j = C * D
    f = delta_j * f
    j += 1

    while abs((delta_j - 1)) >= epsilon:
        D = b + a * D
        if D == 0:
            D = mic
        C = b + (a / C)
        if C == 0:
            C = mic
        D = 1 / D
        delta_j = C * D
        f = delta_j * f
        j += 1


"""
     ____________________________________ TANGENT APPROXIMATION USING POLYNOMIALS _____________________________________
     
     tan(x) = x + (1/3)*(x^3) + (2/15)*(x^5) + (17/315)*(x^7) + (62/2835)*(x^9) <==> tan(x) = x + P(x^2) * (x^3)

"""


def polynomials_approximation(value: float):
    # coefficients:
    c1 = 0.33333333333333333     # 1/3
    c2 = 0.133333333333333333    # 2/15
    c3 = 0.053968253968254       # 17/315
    c4 = 0.0218694885361552      # 62/2835
    return value + c1 * pow(value, 3) + c2 * pow(value, 5) + c3 * pow(value, 7) + c4 * pow(value, 9)
