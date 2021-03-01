import math
import random
import time


pi = math.pi  # 3.141592653589793
pos_inf = float('inf')  # +∞
neg_inf = float('-inf')  # -∞


def bring_in_interval(a):
    """
    :param a: number to be transformed into (-pi/2, pi/2)
    :return: number a in (-pi/2, pi/2) (first quadrant);
             returns x itself, not tan(x)

    we know: tan(x + k*pi) = tan(x)
    """
    k = 0  # periodicity factor

    if a > 0:
        while k * pi <= a:
            k += 1
        return k * pi - a

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


"""
     ________________________________________ THE CONTINUED FRACTIONS METHOD __________________________________________

"""

eps = pow(10, 2)


def calculate_tangent_lorentz(val):
    """
    b[0] = x/(1+)
    a[1], a[2], ..., a[n] = -x^2
    b[1] = 1+
    b[2] = 3+ etc.
    """
    global eps
    b = 1
    f = b
    a = val
    error = pow(10, -13)
    if f == 0:
        f = error  # epsilon = 10^m, m = -13
    running = True
    c = f
    d = 0
    j = 1

    # simulating a do-while:

    d = b + a * d
    if d == 0:
        d = error
    c = b + a / c
    if c == 0:
        c = error
    d = pow(d, -1)
    delta = c * d
    f *= delta
    j += 1

    while running:
        d = b + a * d
        if d == 0:
            d = error
        c = b + a / c
        if c == 0:
            c = error
        d = pow(d, -1)
        delta = c * d
        f *= delta
        j += 1
        if abs(delta - 1) < eps:
            running = False
    return f


"""
     ____________________________________ TANGENT APPROXIMATION USING POLYNOMIALS _____________________________________
     
     tan(x) = x + (1/3)*(x^3) + (2/15)*(x^5) + (17/315)*(x^7) + (62/2835)*(x^9) <==> tan(x) = x + P(x^2) * (x^3)

"""


def compute_tan(val):
    """
    :param val: number for which we compute the tangent
    :return: the tangent of val
    """
    return math.tan(val)


def polynomials_approximation(value):
    # coefficients:
    c1 = 0.33333333333333333  # 1/3
    c2 = 0.133333333333333333  # 2/15
    c3 = 0.053968253968254  # 17/315
    c4 = 0.0218694885361552  # 62/2835
    return value + c1 * pow(value, 3) + c2 * pow(value, 5) + c3 * pow(value, 7) + c4 * pow(value, 9)


def compute_error(a, b):
    """

    :param a: a will be our computed tangent
    :param b: computer's computed tangent
    :return: |a-b|
    """
    return abs(a - b)


print('Generate numbers automatically? (Y/N)')
answer = input()
x = []
if answer == 'Y':
    x = generate_no_in_range()
elif answer == 'N':
    for _ in range(0, 10):  # should change to 10000
        number = input()
        x.append(number)
medium_error_cfm, medium_error_pol = 0, 0
# calculating running time for all 10000:
start_time = time.time()
for i in range(0, len(x)):
    a = x[i]
    print(f'x[{i}] =', x[i], ', '
          f'Continued fraction tan:[{calculate_tangent_lorentz(a)}],',
          f'Polynomial tan: [{polynomials_approximation(a)}],'
          f'Calculator tan: [{compute_tan(a)}],'
          f'CF error: [{compute_error(calculate_tangent_lorentz(a), compute_tan(a))}],'
          f'P error: [{compute_error(polynomials_approximation(a), compute_tan(a))}]')
    medium_error_cfm += compute_error(calculate_tangent_lorentz(a), compute_tan(a))
    medium_error_pol += compute_error(polynomials_approximation(a), compute_tan(a))
# medium --> divided by size of x:
medium_error_pol /= len(x)
medium_error_cfm /= len(x)
print('Medium error for the continued fractions method: ', medium_error_cfm)
print('Medium error for the polynomials approximation method: ', medium_error_pol)
print('Computing time (s): ', time.time() - start_time)
