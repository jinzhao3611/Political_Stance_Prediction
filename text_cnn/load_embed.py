import fasttext
import numpy as np


def load_fasttext(embed_path, vocab_index, dim):
    embed_weights = []
    sorted_vocab_index = sorted(vocab_index.items(), key=lambda k: k[1])
    wv = fasttext.load_model(embed_path)
    for word, idx in sorted_vocab_index:
        if wv.get_word_vector(word) is not None:
            embed_weights.append(wv.get_word_vector(word))
        else:
            embed_weights.append(np.random.uniform(-1, 1, size=dim))

    embed_weights = np.array(embed_weights)
    return embed_weights


def load_fasttext_text(embed_path, vocab_index, dim):
    embed_weights_dict = {}
    embed_weights = []
    sorted_vocab_index = sorted(vocab_index.items(), key=lambda k: k[1])
    vocab = set(vocab_index.keys())
    wv = open(embed_path, "r")
    print(next(wv))
    for vec in wv:
        vec = vec.strip().split()
        if " ".join(vec[:-dim]) in vocab:
            embed_weights_dict[" ".join(vec[:-dim])] = [float(v) for v in vec[-dim:]]
    for word, idx in sorted_vocab_index:
        if embed_weights_dict.get(word) is not None:
            embed_weights.append(embed_weights_dict.get(word))
        else:
            embed_weights.append(np.random.uniform(-1, 1, size=dim))
    embed_weights = np.array(embed_weights)
    return embed_weights
