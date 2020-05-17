import random
from typing import (
    Tuple,
    Optional,
    Generator,
    Union,
    overload,
    List,
    Iterable,
    TypeVar,
    Any,
    Iterator,
    NamedTuple,
    Dict,
)

import attr
import pandas as pd


def _to_tokens(items: Iterable[str]) -> Tuple["Token", ...]:
    return tuple(Token.from_text(item) for item in items)


class Token(NamedTuple):
    text: str
    is_special: bool
    is_hashtag: bool
    is_mention: bool
    is_url: bool

    @staticmethod
    def from_text(text: str) -> "Token":
        is_hashtag = text.startswith("#")
        is_mention = text.startswith("@") or text.startswith(".@")
        is_url = text.startswith("http")

        is_special = any([is_hashtag, is_mention, is_url])
        return Token(text, is_special, is_hashtag, is_mention, is_url,)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


@attr.s(frozen=True, slots=True)
class Sentence(Iterable[Tuple[Token, str]]):
    id: int = attr.ib()
    stance: str = attr.ib()
    tokens: Tuple[Token, ...] = attr.ib()
    tags: Tuple[str, ...] = attr.ib()

    @staticmethod
    def create(
        instance_id: int, stance: str, tokens: List[str], tags: List[str]
    ) -> "Sentence":
        return Sentence(instance_id, stance, _to_tokens(tokens), tuple(tags))

    @property
    def token_strs(self) -> Tuple[str, ...]:
        return tuple(token.text for token in self.tokens)

    def __iter__(self) -> Iterator[Tuple[Token, str]]:
        yield from zip(self.tokens, self.tags)

    @overload
    def __getitem__(self, i: int) -> Tuple[Token, str]:
        raise NotImplementedError

    @overload
    def __getitem__(self, i: slice) -> Tuple[Tuple[Token, ...], Tuple[str, ...]]:
        raise NotImplementedError

    def __getitem__(
        self, i: Union[int, slice]
    ) -> Tuple[Union[Token, Tuple[Token, ...]], Union[str, Tuple[str, ...]]]:
        return self.tokens[i], self.tags[i]

    def copy_with_stance(self, stance: str) -> "Sentence":
        return Sentence(self.id, stance, self.tokens, self.tags)


@attr.s(frozen=True)
class Corpus(Iterable[Sentence]):
    instances: Tuple[Sentence, ...] = attr.ib()

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self.instances)

    @overload
    def __getitem__(self, i: int) -> Sentence:
        raise NotImplementedError

    @overload
    def __getitem__(self, i: slice) -> "Corpus":
        raise NotImplementedError

    def __getitem__(self, i: Union[int, slice]) -> Union[Sentence, "Corpus"]:
        if isinstance(i, slice):
            return Corpus(self.instances[i])
        else:
            return self.instances[i]

    @property
    def stances(self) -> Generator[str, None, None]:
        return (instance.stance for instance in self.instances)

    def shuffled(self, seed: Optional[int]) -> "Corpus":
        if seed is not None:
            random.seed(seed)

        insts = tuple(self.instances)
        random.shuffle(insts)
        return Corpus(insts)


def load_corpus_csv(file: pd.DataFrame, bias_only, addi_mapping) -> Corpus:
    return Corpus(tuple(_gen_instances(file, bias_only, None, addi_mapping)))


def load_corpus(path: str, bias_only: bool, addi_mapping: Optional[Dict]) -> Corpus:

    if path.endswith(".csv"):
        return load_corpus_csv(pd.read_csv(path), bias_only, addi_mapping)
    else:
        raise ValueError(f"File should be of CSV format!")


def _gen_instances(
    file: pd.DataFrame, bias_only: bool, tokenizer: Any, addi_mapping: Dict
) -> Generator[Sentence, None, None]:
    file_data = file.values
    for idx, line in enumerate(file_data):
        if tokenizer:
            tokens = tokenizer(line[0])
        else:
            tokens = line[0].split()
        if addi_mapping:
            addi_tokens = []
            for token in tokens:
                token = token.lower()
                for key in addi_mapping:
                    if key.endswith("*"):
                        if token.startswith(key[:-1]):
                            addi_tokens.extend(addi_mapping[key])
                    else:
                        if token == key:
                            addi_tokens.extend(addi_mapping[key])
            tokens.extend(addi_tokens)
        tags = ["Other"] * len(
            tokens
        )  # in the future we might use token level annotation
        stance = line[1]
        if bias_only and stance == "Cannot-decide":
            continue
        yield Sentence.create(idx, stance, tokens, tags)


if __name__ == "__main__":
    # with open("../data/moral_dict.pkl", "rb") as f:
    #     moral_dict_mapping = pickle.load(f)
    cp = load_corpus(
        "../data/gold-split/test/combined.csv", bias_only=False, addi_mapping=None
    )
