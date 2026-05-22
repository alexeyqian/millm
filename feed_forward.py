import numpy as np

class FeedForward:
	def __init__(self, embed_dim, hidden_dim):
		self.W1 = np.random.randn(embed_dim, hidden_dim)
		self.b1 = np.zeros(hidden_dim)
		self.W2 = np.random.randn(hidden_dim, embed_dim)
		self.b2 = np.zeros(embed_dim)

	def relu(self, x):
		return np.maximum(0, x)

	def forward(self, x):
		z = x@self.W1 + self.b1
		z = self.relu(z)
		y = z@self.W2 + self.b2
		return y

