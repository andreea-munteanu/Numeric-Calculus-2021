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
print("Addition is associative: ", add_is_associative(1.0, u/10, u/10))


""" Finding an example for which multiply_c (×c) is non-associative: """


def multiply_c(a, b):
    return a * b


def mul_is_associative(a, b, c) -> bool:
    return multiply_c(multiply_c(a, b), c) == multiply_c(a, multiply_c(b, c))


# # for a, b, c given, multiplication is associative:
# print("Multiplication is associative for 1.0, u/10, u/10: ", mul_is_associative(1.0, u/10, u/10))


""" In mathematics, multiplication is associative. In computer science, however, the multiplication isn't necessarily.
This is due to the rounding errors that occur during computation. 
Consequently, in order to find a series of 3 numbers that make the multiplication non-associative, we must look 
at 3 numbers of different sizes. 

We're looking at the following example:
u = find_u()
a = u^(-15)
b = u
c = u*10000000
In this case, the multiplication fails to be associative (in computer science only!).
"""
print("Multiplication is associative: ", mul_is_associative(pow(u, -15), u, u*10000000))  # returns False







