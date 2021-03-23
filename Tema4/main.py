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


def gauss_seidel():
    pass


if __name__ == '__main__':
    # extracting data from input files a_i:
    for i in range(1, 6):
        n, p, q, a, b, c = extract_data_from_a(f'a{i}.txt')
        # computation error epsilon = 10 ^ (-p)
        eps = 10 ** (-p)
        # we only move forward if the main diagonal contains non-zero values exclusively:
        if check_diagonal(a, eps):
            f = extract_f(f'f{i}.txt')
            # print(n, p, q, len(a), len(b), len(c), len(f))
        else:
            print("Main diagonal has 0 values. "
                  "The system cannot be solved using successive over-relaxation iterative method.")







