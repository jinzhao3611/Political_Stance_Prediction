import datetime
import logging
import torch
import pandas as pd
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy

data_path = "data/gold-split/bert_data"
label_path = "data/gold-split/label"
val_file = "dev_combined.csv"
tokenizer = "bert-base-uncased"
pretrained_path = "bert-base-uncased"
out_path = "train-dev"

epochs = 4
lr = 6e-5


torch.cuda.empty_cache()

run_start_time = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
logfile = str("log/log-{}-{}.txt".format(run_start_time, "bert_base_uncased"))

logger = logging.getLogger()

device_cuda = torch.device("cuda")
metrics = [{"name": "Acc", "function": accuracy}]


databunch = BertDataBunch(
    data_path,
    label_path,
    tokenizer=tokenizer,
    train_file="train_combined.csv",
    val_file=val_file,
    label_file="labels.csv",
    text_col="text",
    label_col="label",
    batch_size_per_gpu=32,
    max_seq_length=128,
    multi_gpu=False,
    multi_label=False,
    model_type="bert",
)


learner = BertLearner.from_pretrained_model(
    databunch,
    pretrained_path=pretrained_path,
    metrics=metrics,
    device=device_cuda,
    logger=logger,
    output_dir=out_path,
    finetuned_wgts_path=None,
    warmup_steps=200,
    multi_gpu=False,
    is_fp16=True,
    fp16_opt_level="O2",
    multi_label=False,
    logging_steps=100,
)


learner.fit(
    epochs=epochs,
    lr=lr,
    validate=True,
    schedule_type="warmup_cosine",
    optimizer_type="lamb",
)


test_file = "data/gold-split/bert_data/data/dev_combined.csv"
test_csv = pd.read_csv(test_file)
test_data = test_csv["text"].values.tolist()
multiple_predictions = learner.predict_batch(test_data)
pred_labels = [pred[0][0] for pred in multiple_predictions]
test_labels = test_csv["label"].values.tolist()
with open("bert.pred", "w") as f:
    f.writelines("\n".join(pred_labels))
learner.save_model()
