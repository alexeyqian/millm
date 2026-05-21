import math
def dot_product(a, b):
    result = 0
    for x, y in zip(a,b):
        result += x*y
    return result

def magnitude(v):
    total = 0
    for x in v:
        total += x*x
    return math.sqrt(total)

def cosine_similarity(a, b):
    return dot_product(a,b)/(magnitude(a)*magnitude(b))

def matmul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Invalid matrix dimension")
    results = []
    for i in range(rows_A):
        row = []
        for j in range(cols_B):
            total = 0
            for k in range(cols_A):
                total += A[i][k] * B[k][j]
            row.append(total)
        results.append(row)
    return results

def shape(matrix):
    return (len(matrix), len(matrix[0])

