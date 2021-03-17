A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
for i in A:
    print(i)
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
for i in AA:
    print(i)
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
            print("value at position (", row_count, ", ", tup[0], ") is ", tup[1])
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
            print("value at position (", row_count, ", ", tup[0], ") is ", tup[1])
            # output: (toate valorile de pe linia 3):
            # value at position(3, 2) is 1.3
            # value at position(3, 4) is 101.3

          
