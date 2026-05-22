import numpy as np

class TransformerBlock:
	def __init__(self, embed_dim, hidden_dim):
		self.attn = MultiHeadAttention(embed_dim)
		self.norm1 = LayerNorm(embed_dim)
		self.ffn = FeedForward(embed_dim, hidden_dim)
		self.norm2 = LayerNorm(embed_dim)

	def forward(self, x):
		attn_out = self.attn.forward(x)
		x += attn_out
		x = self.norm1.forward(x)

		ffn_out = self.ffn.forward(x)
		x += ffn_out
		x = self.norm2.forward(x)

		return x

