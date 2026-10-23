import numpy as np

# Generate the constraint matrix for the first leakage point of Karatsuba
def gen_hd_matrix(list1, list2):
    matrix = np.zeros((len(list1), len(list2)), dtype=int)

    for i, val1 in enumerate(list1):
        for j, val2 in enumerate(list2):
            if val1 == val2:
                matrix[i, j] = 1

    return matrix

# Generate the constraint matrix for the second leakage point of Karatsuba
def gen_xor_matrix(list1, list2, list3):
    result_matrix = np.zeros((2, 2, 2), dtype=int)

    for i in range(2):
        for j in range(2):
            xor_result = list1[i] ^ list2[j]

            if xor_result in list3:
                index_in_list3 = list3.index(xor_result)
                result_matrix[i, j, index_in_list3] = 1
            else:
                result_matrix[i, j, :] = 0

    return result_matrix

a = [0, 1]
b = [0, 1]
ta = [0, 1]

data = [gen_hd_matrix(a, b), gen_xor_matrix(a, b, ta)]
