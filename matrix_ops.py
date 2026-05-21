import math
import random
from vector_ops import relu, softmax

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

def relu_matrix(X):
    result = []
    for x in X:
        result.append(relu(x))
    return result

def softmax_matrix(logit_matrix):
    prob_matrix = []
    for logits in logit_matrix:
	prob_matrix.append(softmax(logits))
    return prob_matrix
