class Vocabulary:
    def __init__(self):
        self.word_to_id={}
        self.id_to_word={}
    def build(self, text):
        words = text.split()
        words = sorted(set(words))
        for idx, word in enumerate(words):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
    def encode(self, text):
        words = text.split()
        return [self.word_to_id[word] for word in words]
    def decode(self, ids):
        return " ".join(self.id_to_word[i] for i in ids]

