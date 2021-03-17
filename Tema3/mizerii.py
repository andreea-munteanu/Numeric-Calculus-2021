A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
# for i in A:
#     print(i)
# output:
# [1, 2, 3]
# [4, 5, 6]
# [7, 8, 9]

AA = [[(1, 102.5), (3, 2.5)],
      [(1, 3.5), (2, 104.88), (3, 1.05), (5, 0.33)],
      [(3, 100.0)],
      [(2, 1.3), (4, 101.3)],
      [(1, 0.73), (4, 1.5), (5, 102.23)]
      ]
# for i in AA:
#     print(i)
# output:
# [(1, 102.5), (3, 2.5)]
# [(1, 3.5), (2, 104.88), (3, 1.05), (5, 0.33)]
# [(3, 100.0)]
# [(2, 1.3), (4, 101.3)]
# [(1, 0.73), (4, 1.5), (5, 102.23)]

print("Accesarea elementelor de pe coloana 1")
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

print("Accesarea elementelor de pe linia 3")
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

print("This is the number of lines in the file")
print(Counter)
# check that A is written correctly from a.txt
# our sample input file: our_sample.txt
f = open('our_A.txt', "r")
n = int(f.readline())  # matrix size
line = f.readline()
print("n =", n)
B = [[] for _ in range(n)]
print(B)
count = -1
for _ in range(Counter - 1):  # for line in a.txt:
    count += 1
    line_i = f.readline().split(', ')
    val = float(line_i[0])
    row, col = int(line_i[1]), int(line_i[2])
    print(count, val, row, col)
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
        print(row_count, ": ", i)
        # 0: [(0, 102.5), (2, 2.5)]
        # 1: [(0, 3.5), (1, 104.88), (2, 1.05), (4, 0.33)]
        # 2: [(2, 100.0)]
        # 3: [(1, 1.3), (3, 101.3)]
        # 4: [(0, 0.73), (3, 1.5), (4, 102.23)]


a, b, c, = [], [], []


def read_B_from_file(file, a, b, c):
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


read_B_from_file('our_b.txt', a, b, c)

