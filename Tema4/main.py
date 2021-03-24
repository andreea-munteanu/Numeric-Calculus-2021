from math import sqrt
import numpy as np
from numpy import linalg
from scipy.sparse import diags


def extract_data_from_a(file):
    """
    Method for extracting input for tridiagonal matrix from input file 'a_i'.

    :param file: input file 'a_i'
    :return: p, q, a, b, c
    """
    a, b, c = [], [], []
    f = open(file, "r")
    n = int(f.readline())
    p, q = int(f.readline()), int(f.readline())
    empty_line = f.readline()

    for _ in range(0, n):
        val = f.readline()
        a.append(float(val))

    empty_line = f.readline()
    for _ in range(0, n - q):
        val = f.readline()
        b.append(float(val))

    empty_line = f.readline()
    for _ in range(0, n - p):
        val = f.readline()
        c.append(float(val))
    return n, p, q, a, b, c


def extract_f(name):
    """
    Method for extracting input for tridiagonal matrix from input file 'f_i'.

    :param name: input file 'f_i'
    :return: f ∈ R ** n
    """
    f = []
    file = open(name, "r")
    n = int(file.readline())
    empty_line = file.readline()
    for _ in range(0, n):
        val = file.readline()
        f.append(float(val))
    return f


def check_diagonal(diagonal, epsilon) -> bool:
    """
    Method for checking whether the main diagonal (represented as a vector) has only non-zero values.
    We consider non-zero value any a(i, i) for which |a(i, i)| > ε, ∀i = 1..n.

    :param diagonal: main diagonal (as vector)
    :param epsilon: computation error
    :return: true if diagonal contains only non-zero values, false otherwise
    """
    for elem in diagonal:
        if abs(elem) <= epsilon:
            return False
    return True


def gauss_seidel(a, b, c, f, epsilon):
    """
    Method for computing the Gauss Seidel iterative method for sparse linear system solution approximation.

    :return: x_gs
    """
    # initially, xgs = 0:
    xgs = [0 for _ in range(n)]
    delta_x = 0.0
    k = 0
    kmax = 10000  # maximum 10000 iterations
    running = True
    while k < kmax and running:
        # print(k, ": ", sep='', end='\n')
        delta_x = 0.0
        for i in range(0, n):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(1, i):
                sum1 += c[j] * xgs[j]
            for j in range(i+1, n - 1):
                sum2 += b[j] * xgs[j]
            i_xgs = xgs[i]
            xgs[i] = (f[i] - sum1 - sum2) / a[i]
            delta_x += (xgs[i] - i_xgs) ** 2  # delta_x = ||x_c - x_p||
        delta_x = sqrt(delta_x)
        # if we reach convergence, we display the number of iterations:
        if delta_x < epsilon:
            print(f'\nNumber of iterations: {k}')
            running = False
        elif delta_x > 10 ** 8:
            running = False
        else:
            k += 1
    if delta_x >= epsilon:
        raise Exception("Divergence.")
    else:
        # x_c is the approximation of the exact solution
        return xgs


def check_solution(x_gs, f, a, b, c, p, q, n):
    """
    EROARE AICI APARENT DIN CAUZA MEMORIEI


    Method for determining whether our computed solution 'sol' is correct.
    We will compute the norm ||A * x_GS - f||∞.

    :param x_gs: computed solution
    :param a: main diagonal
    :param b: diagonal starting at q
    :param c: diagonal starting at p
    :param p: index p
    :param q: index q
    :return: true if sol is correct with error epsilon, false otherwise
    """
    a = [1, 2, 3, 4, 5]
    b = [5, 6, 7, 8]
    c = [9, 10, 11, 12]
    x_gs = [100, 200, 300, 400, 500]
    prod = []  # A * x_GS
    for i in range(0, n):
        print("i=", i)
        p_i = 0
        if i == 0:
            p_i += a[0] * x_gs[0] + b[0] * x_gs[1]
        elif i == n - 1:
            p_i += c[n-2] * x_gs[n-2] + a[n-1] * x_gs[n-1]
        else:
            p_i += a[i] * x_gs[i]
            # p_i += c[i-1] * x_gs[i-1]
            # p_i += b[i-1] * x_gs[i+1]
        print(p_i)
        prod.append(p_i - f[i])
    print(prod)
    # diagonals = diags(diagonals, [0, p, -q]).toarray()
    # prod = diagonals.dot(x_gs)       # A * x_GS
    # res = prod.subtract(f)           # A * x_GS - f
    return linalg.norm(prod, np.inf)   # ||A * x_GS - f||∞


if __name__ == '__main__':
    # extracting data from input files a_i:
    for i in range(1, 6):
        print(f'\nRun for files a{i}, f{i}:\n_____________________')
        n, p, q, a, b, c = extract_data_from_a(f'a{i}.txt')
        # computation error epsilon = 10 ^ (-p)
        eps = 10 ** (-p)
        # only move forward if main diagonal contains non-zero values only:
        if check_diagonal(a, eps):
            f = extract_f(f'f{i}.txt')
            x_GS = gauss_seidel(a, b, c, f, eps)
            # check_solution(x_GS, f, a, b, c, p, q, n)
            print(f'Checking solution: {check_solution(x_GS, f, a, b, c, p, q, n)}')
            # print("x_GS: ", x_GS, end='\n')
        else:
            print("Main diagonal has 0 values. "
                  "The system cannot be solved using successive over-relaxation iterative method.", end='\n')

