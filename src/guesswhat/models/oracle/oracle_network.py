import tensorflow as tf

from generic.tf_factory.image_factory import get_image_features
from generic.tf_utils.abstract_network import ResnetModel
from neural_toolbox import rnn, utils

import tensorflow.contrib.layers as tfc_layers

class OracleNetwork(ResnetModel):

    def __init__(self, config, no_words, no_answers, device='', reuse=False):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            embeddings = []
            self.batch_size = None

            # QUESTION
            self._is_training = tf.placeholder(tf.bool, name="is_training")
            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=no_words,
                embed_dim=config["model"]["question"]["word_embedding_dim"],
                scope="word_embedding",
                reuse=reuse)


            lstm_states, _ = rnn.variable_length_LSTM(word_emb,
                                                      num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                      seq_length=self._seq_length)
            embeddings.append(lstm_states)

            # CATEGORY
            if config['inputs']['category']:
                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

                cat_emb = tfc_layers.embed_sequence(
                    ids=self._category,
                    vocab_size=config['model']['category']["n_categories"] + 1,
                    embed_dim=config['model']['category']["embedding_dim"],
                    scope="cat_embedding",
                    reuse=reuse)

                embeddings.append(cat_emb)
                print("Input: Category")

            # SPATIAL
            if config['inputs']['spatial']:
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                embeddings.append(self._spatial)
                print("Input: Spatial")


            # IMAGE
            if config['inputs']['image']:
                self._image = tf.placeholder(tf.float32, [self.batch_size] + config['model']['image']["dim"], name='image')
                self.image_out = get_image_features(
                    image=self._image, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['model']['image']
                )
                embeddings.append(self.image_out)
                print("Input: Image")

                self._mask = tf.placeholder(tf.float32, self.image_out.get_shape(), name='mask')

            # CROP
            if config['inputs']['crop']:
                self._crop = tf.placeholder(tf.float32, [self.batch_size] + config['model']['crop']["dim"], name='crop')
                self.crop_out = get_image_features(
                    image=self._crop, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config["model"]['crop'])

                embeddings.append(self.crop_out)
                print("Input: Crop")


            # Compute the final embedding
            emb = tf.concat(embeddings, axis=1)

            # OUTPUT
            num_classes = 3
            self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')

            with tf.variable_scope('mlp'):
                num_hiddens = config['model']['MLP']['num_hiddens']
                l1 = utils.fully_connected(emb, num_hiddens, activation='relu', scope='l1')

                self.out = utils.fully_connected(l1, num_classes) # no need to compute the softmax

                #####################
                #   Loss
                #####################
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self._answer, name='cross_entropy')
                self.loss = tf.reduce_mean(self.cross_entropy)

                self.softmax = tf.nn.softmax(self.out, name='answer_prob')
                self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')


            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            print('Model... Oracle build!')

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy
