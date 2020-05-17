import argparse
import pickle
from math import log
from typing import Dict, List
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from reader.data import load_corpus, Corpus, Sentence
from reader.scoring import score_corpus


def train_svm(train, test):
    params_header = ["c", "class_weight", "feature_func"]
    c_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    class_weight_values = [None, "balanced"]
    feature_funcs = [binary_bow, count_bow, log_count_bow]

    first_row = True
    for c_value in c_values:
        for class_weight in class_weight_values:
            for feature_func in feature_funcs:
                row = [str(c_value), str(class_weight), feature_func.__name__]
                model = MLModel(
                    LinearSVC(C=c_value, class_weight=class_weight, max_iter=10000),
                    feature_func,
                )
                model.train(train)

                pred_corpus = model.predict(test)
                score = score_corpus(test, pred_corpus)
                if first_row:
                    print(",".join(params_header + score.header()))
                    first_row = False
                print(",".join(row + score.row()))


def train_logistic_regression(train, test):
    params_header = ["c", "solver", "feature_func"]

    c_values = [0.05]
    solvers = ["saga"]
    feature_funcs = [binary_bow]

    first_row = True
    for c_value in c_values:
        for solver in solvers:
            for feature_func in feature_funcs:
                row = [str(c_value), solver, feature_func.__name__]
                model = MLModel(
                    LogisticRegression(
                        C=c_value,
                        solver=solver,
                        multi_class="multinomial",
                        penalty="l2",
                        max_iter=10000,
                    ),
                    feature_func,
                )
                model.train(train)

                pred_corpus = model.predict(test)
                score = score_corpus(test, pred_corpus)
                if first_row:
                    print(",".join(params_header + score.header()))
                    first_row = False
                print(",".join(row + score.row()))


def train_MLP(train, test):
    params_header = ["hiddens", "lr", "feature_func"]
    lrs = [0.01, 0.005]
    feature_funcs = [binary_bow, count_bow, log_count_bow]

    first_row = True
    for lr in lrs:
        for feature_func in feature_funcs:
            row = [str(lr), feature_func.__name__]
            model = MLModel(
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    solver="adam",
                    batch_size=32,
                    learning_rate="adaptive",
                    learning_rate_init=lr,
                ),
                feature_func,
            )
            model.train(train)

            pred_corpus = model.predict(test)
            score = score_corpus(test, pred_corpus)
            if first_row:
                print(",".join(params_header + score.header()))
                first_row = False
            print(",".join(row + score.row()))


def train_test(
    train_path: str, test_path: str, bias_only: bool, model_fn, dict_mapping
) -> None:
    train_corpus = load_corpus(train_path, bias_only, dict_mapping)
    test_corpus = load_corpus(test_path, bias_only, dict_mapping)
    model_fn(train_corpus, test_corpus)


def get_feature_names(instance: Sentence, use_bigrams: bool = True, use_trigrams: bool = False) -> List[str]:
    unigrams = [token.text.lower() for token in instance.tokens]
    if use_bigrams:
        bigrams = ["_".join(pair) for pair in zip(unigrams[:-1], unigrams[1:])]
        unigrams.extend(bigrams)
    if use_trigrams:
        trigrams = ["_".join(pair) for pair in zip(unigrams[:-2], unigrams[1:-1], unigrams[2:])]
        unigrams.extend(trigrams)
    return unigrams


def binary_bow(instance: Sentence) -> Dict[str, float]:
    features = get_feature_names(instance)
    return {feat: 1.0 for feat in features}


def count_bow(instance: Sentence) -> Dict[str, float]:
    features = get_feature_names(instance)
    return {
        token: float(count)
        for token, count in Counter(
            features
        ).items()
    }


def log_count_bow(instance: Sentence) -> Dict[str, float]:
    return {token: log(count) for token, count in count_bow(instance).items()}


class MLModel:
    def __init__(self, model, feature_function,) -> None:
        self.feature_func = feature_function
        self.vectorizer = DictVectorizer()
        self.model = model

    def train(self, corpus: Corpus) -> None:
        features = self.vectorizer.fit_transform(
            self.feature_func(instance) for instance in corpus
        )
        labels = tuple(corpus.stances)
        self.model.fit(features, labels)

    def predict(self, corpus: Corpus) -> Corpus:
        features = self.vectorizer.transform(
            self.feature_func(instance) for instance in corpus
        )
        preds = self.model.predict(features)
        return Corpus(
            tuple(
                instance.copy_with_stance(pred) for instance, pred in zip(corpus, preds)
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("test_path")
    parser.add_argument("model")
    parser.add_argument("-b", "--bias", action="store_true")
    parser.add_argument("-m", "--use_moral", action="store_true")
    models = {"logre": train_logistic_regression, "mlp": train_MLP, "svm": train_svm}
    args = parser.parse_args()
    if args.use_moral:
        with open("data/additional_resources/moral_dict.pkl", "rb") as f:
            moral_dict_mapping = pickle.load(f)
    else:
        moral_dict_mapping = None
    train_test(args.train_path, args.test_path, args.bias, models[args.model], moral_dict_mapping)


if __name__ == "__main__":
    from nltk.metrics import ConfusionMatrix
    gold = "a a a c c d".split()
    pred = "a a c c d d".split()
    cm = ConfusionMatrix(gold, pred)
    print(cm['c', 'a'])
