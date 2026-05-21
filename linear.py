import random
import math

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = []
        for i in range(input_size):
            row = []
            for j in range(output_size):
                row.append(random.uniform(-1,1))
            self.weights.append(row)
        self.bias = [0.0 for _ in range(output_size)]

    def forward(self, X):
        if isinstance(X[0], list):
            return [self._forward_single(row) for row in X]
        else:
            return self._forward_single(row)

    def _forward_single(self, x):
        output = []
	for j in range(len(self.weights[0])):
	    total = 0
	    # dot product
            for i in range(len(x)):
		total += x[i]*self.weights[i][j]
            output.append(total)
       return output
