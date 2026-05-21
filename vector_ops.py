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

def softmax(logits_vec):
    exps = []
    for xi in logits_vec:
        exps.append(math.exp(xi))
    probs = []
    total = sum(exps)
    for exp in exps:
        probs.append(exp/total)
    return probs

# pred = [0.7, 0.2, 0.1]
# target = [1, 0, 0]
# loss = 0.3566749
def cross_entropy(preds, targets):
    loss = 0
    for p,t in zip(preds,targets):
	loss += t*math.log(p)
    return -loss
 
