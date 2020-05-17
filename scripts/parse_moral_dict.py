import pickle
from collections import defaultdict


moral_file_dict = "../data/additional_resources/moral.dic"

with open(moral_file_dict, "r") as f:
    lines = f.readlines()

moral_types = {k: v for k, v in [line.strip().split() for line in lines[1:12]]}

moral_mapping = defaultdict(tuple)

for line in lines[14:]:
    if line.strip():
        pair = line.strip().split()
        moral_mapping[pair[0]] = tuple([moral_types[p] for p in pair[1:]])

with open("../data/additional_resources/moral_dict.pkl", "wb") as f:
    pickle.dump(moral_mapping, f)
