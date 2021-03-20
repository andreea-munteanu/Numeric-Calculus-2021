import numpy as np
import copy
from typeA import n, A
from typeB import p, q, a, b, c

# A = [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
# for i in A:
#     print(i)
# output:
# [1, 2, 3]
# [4, 5, 6]
# [7, 8, 9]

AA = [[(0, 102.5), (2, 2.5)],
      [(0, 3.5), (1, 104.88), (2, 1.05), (4, 0.33)],
      [(2, 100.0)],
      [(1, 1.3), (3, 101.3)],
      [(0, 0.73), (3, 1.5), (4, 102.23)]
      ]
# for i in AA:
#     print(i)
# output:
# [(1, 102.5), (3, 2.5)]
# [(1, 3.5), (2, 104.88), (3, 1.05), (5, 0.33)]
# [(3, 100.0)]
# [(2, 1.3), (4, 101.3)]
# [(1, 0.73), (4, 1.5), (5, 102.23)]

#print("Accesarea elementelor de pe coloana 1")
row_count = -1
for i in AA:
    row_count += 1  # row in matrix
    for tup in i:
        if tup[0] == 1:  # col = 1
            pass
            # print("value at position (", row_count, ", ", tup[0], ") is ", tup[1])
            # output: (toate valorile de pe coloana col)
            # value at position(0, 1) is 102.5
            # value at position(1, 1) is 3.5
            # value at position(4, 1) is 0.73


# print("Accesarea elementelor de pe linia 3")
desired_col = 3
row_count = -1
for i in AA:
    row_count += 1  # row in matrix
    if row_count == desired_col:
        for tup in i:
            pass
            # print("value at position (", row_count, ", ", tup[0], ") is ", tup[1])
            # output: (toate valorile de pe linia 3):
            # value at position(3, 2) is 1.3
            # value at position(3, 4) is 101.3

file = open("our_A.txt", "r")
Counter = 0
# Reading from file
Content = file.read()
CoList = Content.split("\n")
for i in CoList:
    if i:
        Counter += 1


#print("This is the number of lines in the file")
#print(Counter)
# check that A is written correctly from a.txt
# our sample input file: our_sample.txt
f = open('our_A.txt', "r")
n = int(f.readline())  # matrix size
line = f.readline()
print("n =", n)
B = [[] for _ in range(n)]
count = -1
for _ in range(Counter - 1):  # for line in a.txt:
    count += 1
    line_i = f.readline().split(', ')
    val = float(line_i[0])
    row, col = int(line_i[1]), int(line_i[2])
    # print(count, val, row, col)
    row_count = -1
    for i in B:  # 'i' is a list containing tuples (col, val)
        row_count += 1
        if row == row_count:
            found_col = False
            for tup in i:
                # if we reached an already registered (row, col) position:
                if tup[0] == col:
                    found_col = True
                    # we save new val, delete old (col, val) and add new tuple (col, new_val):
                    val += tup[1]
                    i.remove(tup)
                    i.append((col, float(val)))
            # if col doesn't exist on row, we add the tuple (col, val) on row 'i':
            if not found_col:
                i.append((col, float(val)))
        # print(row_count, ": ", i)
        # 0: [(0, 102.5), (2, 2.5)]
        # 1: [(0, 3.5), (1, 104.88), (2, 1.05), (4, 0.33)]
        # 2: [(2, 100.0)]
        # 3: [(1, 1.3), (3, 101.3)]
        # 4: [(0, 0.73), (3, 1.5), (4, 102.23)]


a, b, c, = [], [], []


def read_B_from_file(file, a, b, c):
    a, b, c = [], [], []
    f = open(file, "r")
    n = int(f.readline())                           # 2021
    p, q = int(f.readline()), int(f.readline())     # 1, 1
    empty_line = f.readline()
    counter = 3 * n - p - q
    for _ in range(0, n):
        val = f.readline()
        a.append(float(val))
    print("a =", a)

    empty_line = f.readline()
    for _ in range(0, n - p):
        val = f.readline()
        b.append(float(val))
    print("b =", b)

    empty_line = f.readline()
    for _ in range(0, n - q):
        val = f.readline()
        c.append(float(val))
    print("c =", c)
    return a, b, c


a, b, c = read_B_from_file('our_b.txt', a, b, c)


def A_plus_B(n, A, p, q, a, b, c):
    """
    Method for computing sum A + B of sparse matrices.

    :param n: size of square matrices A and B
    :param A: matrix A stored as list of lists
    :param p: distance between main diagonal a and second diagonal b in matrix B
    :param q: distance between main diagonal a and third diagonal b in matrix B
    :param a: main diagonal in B (stored as list)
    :param b: second diagonal in B (stored as list)
    :param c: third diagonal in B (stored as list)
    :return: matrix (A + B)
    """
    SUM = A  # initially, SUM = A
    print("\nSUM (initially) : ", end='\n')
    for i in SUM:
        print(i)
    # adding elements on main diagonal:
    row = -1  # current row in for
    for i in SUM:
        row += 1  # row count
        found = False
        for col in range(n - 1):
            # we look for every col, see if it appears in tuples and add vectors b where necessary
            for tup in i:
                if col == tup[0]:
                    # val will be sum (A+B)[row][col]
                    val = 0
                    if row == col:  # main diagonal
                        val = tup[1] + a[col]
                    elif col - row == q:  # diagonal q
                        val = tup[1] + b[col]
                    elif row - col == p:  # diagonal p
                        val = tup[1] + c[col]
                    # print(tup, val)
                    print("removing ", tup, ", adding", (val, col), "at", (row, col))
                    i.remove(tup)
                    i.append((col, val))
                    found = True
                    break
        if not found:
            i.append((row, a[row]))
            if row < n - 1:
                i.append((row + q, b[row]))
            if row > 0:
                i.append((row - p, c[row]))

    # sorting elements on row by col
    for i in SUM:
        i.sort(key=lambda tup: tup[0])
    return SUM


def A_plus_B2(n, A, p, q, a, b, c):
    SUM = A  # initially, SUM = A
    # print("\nSUM (initially) : ", end='\n')
    for i in SUM:
        print(i)
    # adding elements on main diagonal:
    row = -1  # current row in for
    for i in SUM:
        row += 1  # row count
        count_col = 0
        tup_count = 0
        while count_col < n and tup_count < len(i):
            if count_col == i[tup_count][0]:
                added_val = 0
                # main diagonal
                if row == count_col:
                    added_val = i[tup_count][1] + a[count_col]
                # diagonal b
                elif count_col - row == q:
                    added_val = i[tup_count][1] + b[count_col]
                # diagonal c
                elif row - count_col == p:
                    added_val = i[tup_count][1] + c[count_col]
                # print("removing ", i[tup_count], ", adding", (count_col, added_val), "at", (row, count_col))
                i.remove(i[tup_count])
                i.append((count_col, added_val))
                count_col += 1
                tup_count += 1
            elif count_col < i[tup_count][0]:
                # if row == tup_count:
                #     i.append((tup_count, a[tup_count]))
                # elif tup_count - row == q:
                #     i.append((tup_count, b[tup_count]))
                # elif row - count_col == p:
                #     i.append((tup_count, c[tup_count]))
                count_col += 1
            elif count_col > i[tup_count][0]:
                # if row == count_col:
                #     i.append((count_col, a[count_col]))
                # elif count_col - row == q:
                #     i.append((count_col, b[count_col]))
                # elif row - count_col == p:
                #     i.append((count_col, c[count_col]))
                tup_count += 1
    # sorting elements on row by col
    for i in SUM:
        i.sort(key=lambda tup: tup[0])
    return SUM

#
# print("\nAA: ")
# for i in AA:
#     print(i)


def A_PLUS_B3(n, A, p, q, a, b, c):
    # initially, SUM = A
    SUM = A

    # adding elements on main diagonal (successful)
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] == index:
                found_tup = tup
                found_index = True
        if not found_index:
            row.append((index, a[index]))
        else:
            val = found_tup[1] + a[index]
            row.remove(found_tup)
            if val != 0:
                row.append((index, val))

    # adding elements on diagonal b: (successful)
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] > 0 and tup[0] == index + q:
                found_tup = tup
                found_index = True
        if not found_index:
            if index < n - 1:
                row.append((index + q, b[index]))
        else:
            if index < n:
                val = found_tup[1] + b[index]
                row.remove(found_tup)
                if val != 0:
                    row.append((index + q, val))

    # adding elements on diagonal c: (successful)
    for index, row in enumerate(SUM):
        found_tup = None
        found_index = False
        for tup in row:
            if tup[0] == index - p:
                found_tup = tup
                found_index = True
        if not found_index:
            if index > 0:
                row.append((index - p, c[index - p]))
        else:
            if index < n:
                val = found_tup[1] + c[index - p]
                row.remove(found_tup)
                if val != 0:
                    row.append((index - p, val))

    # sorting elements on row by col
    for i in SUM:
        i.sort(key=lambda tup: tup[0])
    return SUM


APLUSB = A_PLUS_B3(n, AA, 1, 1, a, b, c)
print("\nsum is: ")
for i in APLUSB:
    print(i)


def check_A_plus_B() -> bool:
    pass


# sum = A_plus_B(n, AA, p, q, a, b, c)
# print("\n\nSUM after computation: ")
# for i in sum:
#     print(i)

