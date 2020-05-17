from typing import Tuple
from os import listdir
from os.path import join as pjoin
import pandas as pd
from pandas import DataFrame


def get_sent_length(file_df: "DataFrame") -> Tuple[int, int, int]:
    sentences = file_df["text"].values
    snts_lens = [len(snt.split()) for snt in sentences]
    return max(snts_lens), min(snts_lens), int(sum(snts_lens) / len(snts_lens))


def main():
    data_dir = "../data/raw_data/unsplitted_data"
    files = sorted(listdir(data_dir))
    for file in files:
        print(file, get_sent_length(pd.read_csv(pjoin(data_dir, file))))


if __name__ == '__main__':
    main()