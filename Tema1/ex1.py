""" Sa se gaseasca cel mai mic numar pozitiv u > 0, de forma u = 10^m, astfel ca:
1.0 +c u != 1.0,
unde prin +c am notat operatia de adunare efectuata de calculator.
Numarul u poarta numele de precizia masina"""


def add_c(a, b):
    """
    :param a: number
    :param b: number
    :return: sum of numbers a and b
    """
    return a + b


def check_statement(u) -> bool:
    """
    :param u: u = 10^m
    :return: boolean value: true if 1.0 +c u = 1.0, false otherwise
    """
    return add_c(1.0, u) != 1.0


def find_u():
    """

    :return: minimal machine precision (u) value s.t. 1.0 +c u != 1.0 holds
    """
    power = 1  # m
    machine_precision = None  # u; value to be returned
    running = True
    while running:
        u = pow(10, power)
        if check_statement(u):
            power -= 1
            machine_precision = u
        else:
            running = False
    return machine_precision


print(find_u())
