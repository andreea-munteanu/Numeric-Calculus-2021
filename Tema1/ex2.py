"""
Operatia +c este neasociativa: fie numerele reale a = 1.0, b = u/10, c = u/10,
unde u este precizia masina calculata anterior. Sa se verifice
ca operatia de adunare efectuata de calculator nu este asociativa, i.e.:
(a +c b) +c != a +c (b +c c).

Gasiti un exemplu pentru care operatia ×c este neasociativa
"""

# import necessary functions from ex1.py:
from ex1 import find_u, add_c


def add_is_associative(a, b, c) -> bool:
    """

    :param a: float
    :param b: float
    :param c: float
    :return: boolean value: true if (a +c b) +c = a +c (b +c c) holds, false otherwise
    """
    return add_c(add_c(a, b), c) == add_c(a, add_c(b, c))


u = find_u()
# a = 1.0, b = u/10, c = u/10
print(add_is_associative(1.0, u/10, u/10))










