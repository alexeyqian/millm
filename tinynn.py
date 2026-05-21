import random
import math
import numpy as np

class TinyNN():
    def __init__(self, n_feature, n_classes):
	self.W = np.random.randn(n_features, n_classes)
	self.b = np.zeros((n_features, 1))

    def softmax(self, scores_matrix):
	exp_x = np.exp(scores_matrix)
        row_totals = np.sum(exp_x, axis=1, keepdims=True)
	return exp_x / row_totals

    def forward(self, X):
	logits = X@self.W + self.b
	probs = self.softmax(logits)
	return probs

    def cross_entropy(preds, targets):
	epsilon = 1e-9
	loss = -np.sum(targets*np.log(preds+episilon))
	return loss/preds.shape[0]


