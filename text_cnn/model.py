import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers, initializers


class TextCNN(Model):
    def __init__(
        self, embed, filter_nums, filter_sizes, num_classes, max_sent_len,
    ):
        """

        :param embed: word embeddings
        :param extra_embed:
        :param filter_nums: number of filters
        :param filter_sizes: different sizes of filters
        :param num_classes: number of output classes
        :param max_sent_len: max sentence length
        """
        super(TextCNN, self).__init__()
        self.vocab_size, self.embeddings_dim = embed.shape

        self.max_sent_len = max_sent_len
        self.num_classes = num_classes
        # create the word embedding layer with pre-trained embedding weights, and make it trainable
        self.embed = layers.Embedding(
            self.vocab_size,
            self.embeddings_dim,
            embeddings_initializer=initializers.Constant(embed),
            trainable=True,
        )
        self.concurrent_cnn_layer = []
        self.concurrent_max_pool_layer = []
        self.filter_sizes = filter_sizes
        self.filter_nums = filter_nums
        for filter_size in self.filter_sizes:
            # create one conv and maxpooling layer to handle each one filter size
            self.concurrent_cnn_layer.append(
                layers.Conv2D(
                    data_format="channels_last",
                    filters=filter_nums,
                    kernel_size=(filter_size, self.embeddings_dim),
                    strides=1,
                    padding="valid",
                    activation=tf.nn.relu,
                    kernel_regularizer=regularizers.l2(0.01),
                )
            )
            self.concurrent_max_pool_layer.append(
                layers.MaxPool2D(
                    pool_size=(self.max_sent_len - filter_size + 1, 1),
                    strides=1,
                    padding="valid",
                )
            )
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.out = layers.Dense(
            self.num_classes, kernel_regularizer=regularizers.l2(0.01)
        )

    def call(self, x, use_softmax):
        features = []
        words = self.embed(x)
        x = tf.expand_dims(words, -1)
        for i in range(len(self.filter_sizes)):
            feature = self.concurrent_cnn_layer[i](x)
            feature = self.concurrent_max_pool_layer[i](feature)
            features.append(feature)
        flatted = self.flatten(tf.concat(features, axis=-1))
        flatted = self.dropout(flatted)
        out = self.out(flatted)
        if use_softmax:
            out = tf.nn.softmax(out)
        return out


def cross_entropy_loss(y_pred, y_true):
    y_true = tf.cast(y_true, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


if __name__ == "__main__":
    pass
