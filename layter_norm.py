import numpy as np

class LayerNorm:
	def __init__(selt, dim, eps=1e-5):
		self.gamma = np.ones(dim)
		self.beta = np.zeros(dim)

	def forward(self, x):
		mean = np.mean(x, axis=-1, keepdims=True)
		var = np.var(x, axis=-1, keepdims=True)
		normalized = (x-mean)/np.sqrt(var+self.eps)
		return self.gamma*normalized + self.beta

