from os.path import join as pjoin
from os import listdir
import pandas as pd
from pandas import DataFrame


def get_best_performance(input_df: "DataFrame", binary: bool) -> "DataFrame":
    line = input_df.iloc[input_df["micro_f"].argmax()]
    if binary:
        result = line[["micro_f", "Left-leaning_f", "Right-leaning_f",
                       "Left-leaning_supp", "Right-leaning_supp"]]
    else:
        result = line[["micro_f", "Left-leaning_f", "Right-leaning_f", "Cannot-decide_f",
                       "Left-leaning_supp", "Right-leaning_supp", "Cannot-decide_supp"]]
    return result


def all_categories_results_table(res_dir: str, model_name: str) -> str:
    file_path = pjoin(res_dir, model_name)
    files = sorted(listdir(file_path))
    try:
        row_names = [f.split("_")[0] for f in files]
    except:
        row_names = [f.split("-")[1].split(".")[0] for f in files]
    results = []
    if "binary" in model_name:
        flag = True
    else:
        flag = False
    for file in files:
        results.append(get_best_performance(pd.read_csv(pjoin(file_path, file)), flag))
    res_df = pd.concat(results, axis=1, ignore_index=True).T
    res_df.index = row_names
    return res_df.to_latex()


def main():
    res_dir = "../data/final_scores"
    model_name = "svm_binary"
    latex_t = all_categories_results_table(res_dir, model_name)
    print(latex_t)


if __name__ == '__main__':
    main()
