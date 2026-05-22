import numpy as np
class SelfAttention:
	def __init__(self, embed_dims):
		self.Wq = np.random.randn(embed_dim, embed_dim)
		self.Wk = np.random.randn(embed_dim, embed_dim)
		self.Wv = np.random.randn(embed_dim, embed_dim)
	def softmax(self, x):
		exp_x = np.exp(x-np.max(x, axis=-1, keepdims=True)
		return exp_x/np.sum(exp_x, axis=-1, keepdims=True)

	def forward(self, x):
		Q = x@self.Wq
		K = x@self.Wk
		V = x@self.Wv
		scores = Q@K.T
		scores = scores / np.sqrt(self.embed_dim)
		weights = self.softmax(scores)
		output = weights@V
		return output

class MultiHeadAttention:
	def __init__(self, embed_dims, num_heads):
		self.embed_dims = embed_dims
		self.num_heads = num_heads
		self.head_dim = embed_dims//num_heads
		self.Wq = np.random.randn(embed_dim, embed_dim)
		self.Wk = np.random.randn(embed_dim, embed_dim)
		self.Wv = np.random.randn(embed_dim, embed_dim)
		self.Wo = np.random.randn(embed_dim, embed_dim)
	def softmax(self, x):
		exp_x = np.exp(x-np.max(x, axis=-1, keepdims=True)
		return exp_x/np.sum(exp_x, axis=-1, keepdims=True)
	def split_heads(self, x):
		seq_len = x.shape[0]
		return x.reshape(seq_len, self.num_heads, self.head_dim)
	def forward(self, x):
		Q = x@self.Wq
		K = x@self.Wk
		V = x@self.Wv
		
		Q = self.split_heads(Q)
		K = self.split_heads(K)
		V = self.split_heads(V)
		heads = []
		for h in range(self.num_heads):
			q,k,v = Q[:, h, :], K[:, h, :], V[:, h, :]
			scores = q@k.T
			scores = np.sqrt(self.head_dim)
			attn = self.softmax(scores)
			head = attn@v
			heads.append(head)
		concat = np.concatenate(heads, axis=-1)
		return concat@Wo	
