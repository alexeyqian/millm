class CharTokenizer:
	def __init__(self, text):
		chars = sorted(set(text))
		self.stoi = {ch:i for i, ch in enumerate(chars)}
		self.itos = {i:ch for i, ch in enumerate(chars)}

	def encode(self, text):
		return [self.stoi[c] for c in text]

	def decode(self, ids):
		return "".join([self.itos[id] for id in ids])

