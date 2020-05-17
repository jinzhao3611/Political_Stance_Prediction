import pandas as pd
import os
from collections import defaultdict
import numpy as np
import json

flair_fastText_path = "flair_outputs_fastText/prfs"
flair_glove_path = "flair_outputs_glove/prfs"
flair_glove_upsampled_path = "flair_outputs_glove_upsampled/prfs"
fastText_clf_path = "fastText_clf_outputs/prfs/prfs.txt"
CNN_path = "fastText_clf_outputs/prfs/prfs1.txt"

policies = [
    "healthcare",
    "economic",
    "immigration",
    "education",
    "abortion",
    "LGBTQ",
    "gun",
    "environment",
    "combined",
]

dataframes = list()
flair_fastText_scores = defaultdict(dict)
flair_glove_scores = defaultdict(dict)
flair_glove_upsampled_scores = defaultdict(dict)

scores_list = [flair_glove_upsampled_scores, flair_fastText_scores, flair_glove_scores]
for filename in os.listdir(flair_fastText_path):
    if filename.endswith(".txt"):
        with open(os.path.join(flair_fastText_path, filename), "r") as infile:
            score_info = infile.readlines()
        filename_wzout_ext = os.path.splitext(filename)[0]
        flair_fastText_scores[filename_wzout_ext]["micro_f1"] = score_info[0].rsplit()[-1]
        flair_fastText_scores[filename_wzout_ext]["l_p"] = score_info[4].rsplit()[14]
        flair_fastText_scores[filename_wzout_ext]["l_r"] = score_info[4].rsplit()[17]
        flair_fastText_scores[filename_wzout_ext]["l_f1"] = score_info[4].rsplit()[-1]
        flair_fastText_scores[filename_wzout_ext]["r_p"] = score_info[5].rsplit()[14]
        flair_fastText_scores[filename_wzout_ext]["r_r"] = score_info[5].rsplit()[17]
        flair_fastText_scores[filename_wzout_ext]["r_f1"] = score_info[5].rsplit()[-1]
        flair_fastText_scores[filename_wzout_ext]["u_p"] = score_info[3].rsplit()[14]
        flair_fastText_scores[filename_wzout_ext]["u_r"] = score_info[3].rsplit()[17]
        flair_fastText_scores[filename_wzout_ext]["u_f1"] = score_info[0].rsplit()[-1]
for filename in os.listdir(flair_glove_path):
    if filename.endswith(".txt"):
        with open(os.path.join(flair_glove_path, filename), "r") as infile:
            score_info = infile.readlines()
        filename_wzout_ext = os.path.splitext(filename)[0]
        flair_glove_scores[filename_wzout_ext]["micro_f1"] = score_info[0].rsplit()[-1]
        flair_glove_scores[filename_wzout_ext]["l_p"] = score_info[4].rsplit()[14]
        flair_glove_scores[filename_wzout_ext]["l_r"] = score_info[4].rsplit()[17]
        flair_glove_scores[filename_wzout_ext]["l_f1"] = score_info[4].rsplit()[-1]
        flair_glove_scores[filename_wzout_ext]["r_p"] = score_info[5].rsplit()[14]
        flair_glove_scores[filename_wzout_ext]["r_r"] = score_info[5].rsplit()[17]
        flair_glove_scores[filename_wzout_ext]["r_f1"] = score_info[5].rsplit()[-1]
        flair_glove_scores[filename_wzout_ext]["u_p"] = score_info[3].rsplit()[14]
        flair_glove_scores[filename_wzout_ext]["u_r"] = score_info[3].rsplit()[17]
        flair_glove_scores[filename_wzout_ext]["u_f1"] = score_info[0].rsplit()[-1]
for filename in os.listdir(flair_glove_upsampled_path):
    if filename.endswith(".txt"):
        with open(os.path.join(flair_glove_upsampled_path, filename), "r") as infile:
            score_info = infile.readlines()
        filename_wzout_ext = os.path.splitext(filename)[0]
        flair_glove_upsampled_scores[filename_wzout_ext]["micro_f1"] = score_info[
            0
        ].rsplit()[-1]
        flair_glove_upsampled_scores[filename_wzout_ext]["l_p"] = score_info[4].rsplit()[
            14
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["l_r"] = score_info[4].rsplit()[
            17
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["l_f1"] = score_info[4].rsplit()[
            -1
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["r_p"] = score_info[5].rsplit()[
            14
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["r_r"] = score_info[5].rsplit()[
            17
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["r_f1"] = score_info[5].rsplit()[
            -1
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["u_p"] = score_info[3].rsplit()[
            14
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["u_r"] = score_info[3].rsplit()[
            17
        ]
        flair_glove_upsampled_scores[filename_wzout_ext]["u_f1"] = score_info[0].rsplit()[
            -1
        ]


with open(fastText_clf_path, "r") as infile:
    fastText_data2 = json.load(infile)
with open(CNN_path, "r") as infile:
    fastText_data3 = json.load(infile)

columns = ["l_p", "l_r", "l_f1", "r_p", "r_r", "r_f1", "u_p", "u_r", "u_f1", "micro_f1"]
for policy in policies:
    scores_by_colums = defaultdict(list)
    for col in columns:
        scores_by_colums_values = list()
        for scores in scores_list:
            scores_by_colums_values.append(scores[policy][col])
        scores_by_colums[col] = scores_by_colums_values

    policy_data2 = fastText_data2["{}.csv".format(policy)]
    data_content2 = (
        list(policy_data2["__label__Left-leaning"].values())
        + list(policy_data2["__label__Right-leaning"].values())
        + list(policy_data2["__label__Cannot-decide"].values())
        + [policy_data2["micro-avaraging"][-1]]
    )

    policy_data3 = fastText_data3["{}.csv".format(policy)]
    data_content3 = (
        list(policy_data3["__label__Left-leaning"].values())
        + list(policy_data3["__label__Right-leaning"].values())
        + list(policy_data3["__label__Cannot-decide"].values())
        + [policy_data3["micro-avaraging"][-1]]
    )

    data_content1 = np.array(list(scores_by_colums.values())).T.tolist()

    data_content1.append(data_content2)
    data_content1.append(data_content3)

    df = pd.DataFrame(
        data_content1,
        index=[
            "flair_glove_upsampled_scores",
            "flair_fastText_scores",
            "flair_glove_scores",
            "CNN",
            "fastText_clf",
        ],
        columns=columns,
    )
    dataframes.append(df)


for policy, df in zip(policies, dataframes):
    df.to_csv("models_performance_per_topic/{}.csv".format(policy))
