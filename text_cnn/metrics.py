from decimal import ROUND_HALF_UP, Context
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

rounder = Context(rounding=ROUND_HALF_UP, prec=4)


def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def F1(y_pred, y_true):
    y_pred = tf.argmax(y_pred, 1).numpy()
    return rounder.create_decimal_from_float(f1_score(y_true, y_pred, average="micro"))


def precision(y_pred, y_true):
    y_pred = tf.argmax(y_pred, 1).numpy()
    return rounder.create_decimal_from_float(
        precision_score(y_true, y_pred, average="micro")
    )


def recall(y_pred, y_true):
    y_pred = tf.argmax(y_pred, 1).numpy()
    return rounder.create_decimal_from_float(
        recall_score(y_true, y_pred, average="micro")
    )


def evaluation(y_pred, y_true):
    f1 = F1(y_pred, y_true)
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return prec, rec, f1


if __name__ == "__main__":
    pass
