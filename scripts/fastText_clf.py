import fasttext
import os
import json

model = fasttext.train_supervised(
    input="data/flair_data2/train/combined.csv",
    epoch=25,
    lr=0.5,
    wordNgrams=2,
    bucket=200000,
    dim=50,
    loss="ova",
)
model.save_model("fastText_models/fastText_combined.bin")

# model = fasttext.load_model('fastText_models/fastText_combined.bin')

scores = dict()
input_folder_path = "data/flair_data2/dev/"
for filename in os.listdir(input_folder_path):
    if filename.endswith(".csv"):
        score = model.test_label(os.path.join(input_folder_path, filename))
        score["micro-avaraging"] = model.test(os.path.join(input_folder_path, filename))
        scores[filename] = score


with open("fastText_clf_outputs/prfs1.txt", "w") as jsonfile:
    json.dump(scores, jsonfile, indent=2)
