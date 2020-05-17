import itertools
from typing import NamedTuple, Dict, Sequence, Tuple, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from reader.data import Corpus


class PRF(NamedTuple):
    precision: float
    recall: float
    f_score: float
    support: int

    @staticmethod
    def create(
        true_positives: int, false_positives: int, false_negatives: int, support: int
    ) -> "PRF":
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return PRF(precision, recall, f1(precision, recall), support)


class Score(NamedTuple):
    micro_f_score: float
    macro_f_score: float
    class_prf: Dict[str, PRF]
    accuracy: float
    support: int

    def pretty_str(self) -> str:
        n_correct = int(round(self.accuracy * self.support))
        lines = [
            f"Micro F1: {self.micro_f_score * 100:2.2f}",
            f"Macro F1: {self.macro_f_score * 100:2.2f}",
            f"Accuracy: {self.accuracy * 100:2.2f} ({n_correct} / {self.support})",
            "Classes:",
        ] + [
            f"\t{label}:\t"
            + f"P {prf.precision* 100:2.2f}\tR {prf.recall* 100:2.2f}\tF {prf.f_score* 100:2.2f} "
            + f"({prf.support})"
            for label, prf in self.class_prf.items()
        ]
        return "\n".join(lines)

    def header(self) -> List[str]:
        return list(
            itertools.chain.from_iterable(
                *(
                    [["micro_f", "macro_f", "acc", "supp"]]
                    + [
                        [f"{label}_prec", f"{label}_rec", f"{label}_f", f"{label}_supp"]
                        for label in sorted(self.class_prf.keys())
                    ],
                )
            )
        )

    def row(self) -> List[str]:
        return list(
            itertools.chain.from_iterable(
                *(
                    [
                        [
                            f"{self.micro_f_score * 100:2.2f}",
                            f"{self.macro_f_score * 100:2.2f}",
                            f"{self.accuracy * 100:2.2f}",
                            str(self.support),
                        ]
                    ]
                    + [
                        [
                            f"{prf.precision * 100:2.2f}",
                            f"{prf.recall * 100:2.2f}",
                            f"{prf.f_score * 100:2.2f}",
                            str(prf.support),
                        ]
                        for label, prf in sorted(self.class_prf.items())
                    ],
                )
            )
        )


def f1(precision: float, recall: float) -> float:
    """Compute F1, returning 0.0 if undefined."""
    if precision and recall:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0.0


def score_corpus(gold_corpus: Corpus, pred_corpus: Corpus) -> Score:
    assert len(gold_corpus) == len(pred_corpus), "Corpora of different lengths"
    assert [sentence.id for sentence in gold_corpus] == [
        sentence.id for sentence in pred_corpus
    ], "Sentence IDs do not match"

    gold_labels = [sentence.stance for sentence in gold_corpus]
    pred_labels = [sentence.stance for sentence in pred_corpus]
    labels = sorted(set(gold_labels))
    micro_f1 = PRF(
        *precision_recall_fscore_support(
            gold_labels, pred_labels, average="micro", zero_division=0
        )
    )
    macro_f1 = PRF(
        *precision_recall_fscore_support(
            gold_labels, pred_labels, average="macro", zero_division=0
        )
    )

    label_prf: Dict[str, PRF] = {
        label: prf
        for label, prf in zip(
            labels,
            _parse_prfs(
                precision_recall_fscore_support(
                    gold_labels, pred_labels, labels=labels, zero_division=0
                )
            ),
        )
    }
    accuracy = accuracy_score(gold_labels, pred_labels)

    return Score(
        micro_f1.f_score, macro_f1.f_score, label_prf, accuracy, len(gold_corpus)
    )


def _parse_prfs(
    result: Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[int]]
) -> List[PRF]:
    return [
        PRF(precision, recall, f_score, support)
        for precision, recall, f_score, support in zip(*result)
    ]
