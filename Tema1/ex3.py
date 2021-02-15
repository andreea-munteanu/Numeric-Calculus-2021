import math
import random

pi = math.pi             # 3.141592653589793
pos_inf = float('inf')   # +∞
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
        if -pi / 2 < values[i] < pi / 2:
            pass
        elif -pi / 2 < values[i] > pi / 2:
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


def calculate_tangent():
    pass
