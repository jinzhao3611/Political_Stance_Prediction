import json
from text_cnn.model import TextCNN, cross_entropy_loss
from text_cnn.vocab import Vocabulary, DataTransformer
from text_cnn.metrics import accuracy, F1, precision, recall, evaluation
from text_cnn.load_embed import load_fasttext_text, load_fasttext

with open("text_cnn/params.json", "r") as f:
    config = json.load(f)

cnn_config = config["CNN"]
