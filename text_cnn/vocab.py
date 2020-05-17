import numpy as np
from typing import List

from reader.data import Corpus, Sentence

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class Vocabulary(object):
    def __init__(self, train_corpus: Corpus):
        self.UNK = "<UNK>"
        self.PAD = "<PAD>"

        self.vocab_index = self._get_vocab_index(train_corpus)

    @staticmethod
    def _instance_tokens(instance: Sentence) -> List[str]:
        return [token.text for token in instance.tokens]

    def _get_vocab_index(self, corpus: Corpus):
        self.vocab = set()
        for ins in corpus:
            self.vocab.update(self._instance_tokens(ins))
        vocab_index = {v: i for i, v in enumerate(self.vocab, 2)}
        vocab_index[self.UNK] = 0
        vocab_index[self.PAD] = 1
        return vocab_index

    def add_word(self, word: str):
        if word in self.vocab:
            return
        else:
            self.vocab.add(word)
            self.vocab_index[word] = len(self.vocab_index)


class DataTransformer(object):
    def __init__(self, corpus: Corpus, max_len=25, vocab: Vocabulary = None):
        self.max_len = max_len
        self.label_encoding = {"Cannot-decide": 0, "Left-leaning": 1, "Right-leaning": 2}
        if isinstance(vocab, Vocabulary):
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(corpus)
        data = []
        labels = []
        for ins in corpus:
            data.append(self._instance_tokens(ins))
            labels.append(ins.stance)

        self.data = self._compose(data)
        self.labels = self._label2idx(labels)

    @staticmethod
    def _instance_tokens(instance: Sentence) -> List[str]:
        return [token.text for token in instance.tokens]

    def train_val_split(self, val_size=0.1, shuffle=True, random_state=521):
        X_train, X_test, y_train, y_test = train_test_split(
            self.data,
            self.labels,
            test_size=val_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        return (X_train, y_train), (X_test, y_test)

    def _token2idx(self, tokens: List[str]):
        unk_idx = self.vocab.vocab_index[self.vocab.UNK]
        token_idx = [self.vocab.vocab_index.get(t, unk_idx) for t in tokens]
        return token_idx

    def _label2idx(self, labels: List[str]):
        label_idx = np.array([self.label_encoding[label] for label in labels])
        return label_idx

    def _compose(self, data: List[List[str]]):
        data_lst = []
        for line in data:
            data_lst.append(self._token2idx(line))
        data = self._padding(data_lst)
        return data

    def _padding(self, idx_matrix: List[List[str]]):
        pad_value = self.vocab.vocab_index[self.vocab.PAD]
        padded = pad_sequences(
            idx_matrix, maxlen=self.max_len, padding="post", value=pad_value
        )
        return padded


if __name__ == "__main__":
    pass
