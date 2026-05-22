import numpy as np
class Embedding:
	def __init__(self, vocab_size, embed_dim):
		self.embeddings = np.random.randn(vocab_size, embed_dim)
	def forward(self, token_ids):
		return self.embeddings[token_ids]

class PositionalEmbedding:
	def __init__(self, max_length, embed_dim):
		self.pe = np.random.randn(max_length, embed_dim)
	def forward(self, length):
		return self.pe[:length]

