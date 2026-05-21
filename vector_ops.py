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

def relu(x_vec):
    result = []
    for v in x_vec:
        result.append(max(0, v))
    return result


