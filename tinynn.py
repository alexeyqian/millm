import random
import math
import numpy as np

class TinyNN():
    def __init__(self, n_feature, n_classes, lr=0.01):
	self.W = np.random.randn(n_features, n_classes)
	self.b = np.zeros((n_features, 1))
	self.lr = lr

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

    def backward(self, X, y, probs):
	n_samples = X.shape[0]
	d_logits = (probs - y)/n_samples
	dW = X.T@d_logits
	db = np.sum(d_logits, axis=0, keepdims=True)
	self.W -= self.lr*dW
	self.b -= self.lr*db
