import argparse
from text_cnn import TextCNN, cross_entropy_loss, cnn_config
from text_cnn import Vocabulary, DataTransformer
from text_cnn import accuracy, evaluation, load_fasttext_text
from reader.data import load_corpus, Corpus
from reader.scoring import score_corpus

import tensorflow as tf
import os
import datetime


def get_timestamp():
    return datetime.datetime.today().strftime("%Y%m%d-%H%M")


def train(
    model,
    train_data: DataTransformer,
    dev_data: DataTransformer,
    test_data: DataTransformer,
    out_name,
):
    train_X = train_data.data
    train_y = train_data.labels

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_X, train_y))
        .repeat()
        .shuffle(buffer_size=10000)
    )
    train_dataset = train_dataset.batch(batch_size=cnn_config["batch_size"]).prefetch(
        buffer_size=1
    )

    dev_X = dev_data.data
    dev_y = dev_data.labels

    test_X = test_data.data
    test_y = test_data.labels

    optimizer = tf.optimizers.Adam(cnn_config["lr"])

    def run_optimization(x, y):
        # Wrap computation inside a GradientTape for automatic differentiation
        with tf.GradientTape() as g:
            # Forward pass
            pred = model(x, use_softmax=False)
            # Compute loss
            loss = cross_entropy_loss(pred, y)
        # Variables to update
        trainable_variables = model.trainable_variables
        # Compute gradients
        gradients = g.gradient(loss, trainable_variables)
        # Update W and b following gradients
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    highest_val_f1 = -1
    early_stop = 0
    for step, (batch_x, batch_y) in enumerate(train_dataset.take(cnn_config["steps"]), 1):
        # Run the optimization to update W and b values
        run_optimization(batch_x, batch_y)
        if step % 100 == 0:
            pred = model(batch_x, use_softmax=True)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)

            dev_pred = model(dev_X, use_softmax=True)
            val_pre, val_recall, val_f1 = evaluation(dev_pred, dev_y)

            if step % 100 == 0:
                print(
                    f"step: {step:03} loss: {loss:.6f}\ttrain acc: {acc:.4f}\tF1: {val_f1:.4f}\tP: {val_pre:.4f}\tR: {val_recall:.4f}"
                )

            if val_f1 > highest_val_f1:
                highest_val_f1 = val_f1
                early_stop = 0
                model.save_weights(filepath=out_name)
                test_pred = model(test_X, use_softmax=True)
                pred_out = tf.argmax(test_pred, 1).numpy()
                test_pre, test_recall, test_f1 = evaluation(test_pred, test_y)
                print(
                    f"On Test: F1: {test_f1:.4f}\tP: {test_pre:.4f}\tR: {test_recall:.4f}"
                )
            else:
                early_stop += 1
            if early_stop > 5:
                break


def predict(model, max_len, test_corpus: Corpus, vocab, ckpt_path=None) -> Corpus:
    if ckpt_path is not None:
        model.load_weights(ckpt_path)
    label_encoding = {"Cannot-decide": 0, "Left-leaning": 1, "Right-leaning": 2}
    rev_label_encoding = {v: k for k, v in label_encoding.items()}
    test_data = DataTransformer(test_corpus, max_len=max_len, vocab=vocab)
    pred = model(test_data.data, use_softmax=True)
    pre, rec, f1 = evaluation(pred, test_data.labels)
    print(f"On Test: F1: {f1:.4f}\tP: {pre:.4f}\tR: {rec:.4f}")
    pred = tf.argmax(pred, 1).numpy()
    return Corpus(
        tuple(
            instance.copy_with_stance(rev_label_encoding[pred])
            for instance, pred in zip(test_corpus, pred)
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("dev_path")
    parser.add_argument("test_path")
    parser.add_argument("embed_path")
    args = parser.parse_args()
    MODEL_DIR = "cnn_output"
    train_corpus = load_corpus(args.train_path, bias_only=False, addi_mapping=None)
    dev_corpus = load_corpus(args.dev_path, bias_only=False, addi_mapping=None)
    test_corpus = load_corpus(args.test_path, bias_only=False, addi_mapping=None)

    vocab = Vocabulary(train_corpus)
    embed = load_fasttext_text(args.embed_path, vocab.vocab_index, dim=300)
    fs = [3, 4, 5]
    train_tr = DataTransformer(train_corpus, max_len=30, vocab=vocab)
    dev_tr = DataTransformer(dev_corpus, max_len=30, vocab=vocab)
    test_tr = DataTransformer(test_corpus, max_len=30, vocab=vocab)

    senti_cnn = TextCNN(
        embed=embed, filter_nums=100, filter_sizes=fs, max_sent_len=30, num_classes=3,
    )

    ts = get_timestamp()
    train(
        senti_cnn,
        train_tr,
        dev_tr,
        test_tr,
        out_name=os.path.join(MODEL_DIR, ts, f'cnn_{cnn_config["embed_dim"]}.ckpt'),
    )

    senti_cnn = TextCNN(
        embed=embed, filter_nums=100, filter_sizes=fs, max_sent_len=30, num_classes=3
    )
    pred_corpus = predict(
        senti_cnn,
        30,
        test_corpus,
        os.path.join(MODEL_DIR, ts, f'cnn_{cnn_config["embed_dim"]}.ckpt'),
    )
    score = score_corpus(test_corpus, pred_corpus)
    print(score)


if __name__ == "__main__":
    main()
