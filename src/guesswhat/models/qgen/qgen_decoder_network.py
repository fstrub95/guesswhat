import tensorflow as tf

import tensorflow.contrib.layers as tfc_layers
import tensorflow.contrib.rnn as tfc_rnn
import tensorflow.contrib.seq2seq as tfc_seq

from generic.tf_factory.image_factory import get_image_features
from generic.tf_utils.abstract_network import AbstractNetwork

import neural_toolbox.rnn as rnn

class QGenNetworkDecoder(AbstractNetwork):


    #TODO: add dropout
    def __init__(self, config, num_words, policy_gradient, start_token, stop_token, device='', reuse=False):
        AbstractNetwork.__init__(self, "qgen", device=device)

        # Create the scope for this graph
        with tf.variable_scope(self.scope_name, reuse=reuse):

            # Misc
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            batch_size = None

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))


            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')
            self.image_out = get_image_features(
                image=self._image,
                question=None, # no attention at this point
                is_training=self._is_training,
                scope_name="image_processing",
                config=config['image'])


            #####################
            #   DIALOGUE
            #####################

            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length_dialogue = tf.placeholder(tf.int32, [batch_size], name='seq_length_dialogue')

            with tf.variable_scope('word_embedding', reuse=reuse):
                embedding_encoder = tf.get_variable("embedding_encoder",
                                                    shape=[num_words, config["dialogue"]["word_embedding_dim"]])

            word_emb_dialogue = tf.nn.embedding_lookup(params=embedding_encoder, ids=self._dialogue)

            if config["dialogue"]['glove']:
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb_dialogue = tf.concat([word_emb_dialogue, self._glove], axis=2)

            word_emb_dialogue = tf.nn.dropout(word_emb_dialogue, dropout_keep)
            self.rnn_states, self.last_rnn_state = rnn.gru_factory(
                inputs=word_emb_dialogue,
                seq_length=self._seq_length_dialogue,
                num_hidden=config["dialogue"]["rnn_state_size"],
                bidirectional=config["dialogue"]["bidirectional"],
                max_pool=config["dialogue"]["max_pool"],
                reuse=reuse)
            self.last_rnn_state = tf.nn.dropout(self.last_rnn_state, dropout_keep)


            #####################
            #   TARGET QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [batch_size, None], name='question')
            self._seq_length_question = tf.placeholder(tf.int32, [batch_size], name='seq_length_question')

            word_emb_question = tf.nn.embedding_lookup(params=embedding_encoder, ids=self._question)

            #####################
            #   VIS
            #####################

            with tf.variable_scope('Vis'):

                self.dialogue_projection = tfc_layers.fully_connected(self.last_rnn_state,
                                                                      num_outputs=config["top"]["vis"]["projection_size"],
                                                                      activation_fn=tf.nn.tanh,
                                                                      reuse=reuse,
                                                                      scope="dialogue_projection")

                self.image_projection = tfc_layers.fully_connected(self.image_out,
                                                                      num_outputs=config["top"]["vis"]["projection_size"],
                                                                      activation_fn=tf.nn.tanh,
                                                                      reuse=reuse,
                                                                      scope="image_projection")

                full_projection = self.image_projection * self.dialogue_projection
                full_projection = tf.nn.dropout(full_projection, dropout_keep)

                self.final_embedding = tfc_layers.fully_connected(full_projection,
                                                                  num_outputs=config["top"]["vis"]["embedding_size"],
                                                                  activation_fn=tf.nn.relu,
                                                                  reuse=reuse,
                                                                  scope="final_projection")
                self.final_embedding = tf.nn.dropout(self.final_embedding, dropout_keep)


            #####################
            #   DECODER
            #####################

            decoder_cell =  tfc_rnn.GRUCell(num_units=config["top"]["decoder"]["num_units"],
                                            activation=tf.nn.tanh,
                                            reuse=reuse)


            training_helper = tfc_seq.TrainingHelper(inputs=word_emb_question, #  The question is the target
                                                     sequence_length=self._seq_length_question)

            projection_layer = tf.layers.Dense(num_words, use_bias=False)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, training_helper, self.final_embedding,
                output_layer=projection_layer)

            decoder_outputs, _ = tfc_seq.dynamic_decode(decoder,
                                                        maximum_iterations=config["top"]["decoder"]["maximum_iterations"])

            tfc_seq.GreedyEmbeddingHelper(
                word_emb_dialogue,
                tf.fill([batch_size], start_token), stop_token)


            #####################
            #   LOSS
            #####################

            # compute the softmax for evaluation
            with tf.variable_scope('decoder_output'):
                self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_outputs, labels=self._question)

                if not policy_gradient:
                    self.loss = tf.reduce_mean(self.cross_entropy_loss)


    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.loss



    #TODO build graph
    # decoder = tf.contrib.seq2seq.BasicDecoder(
    #     decoder_cell, training_helper, self.final_embedding,
    #     output_layer=projection_layer)


if __name__ == "__main__":

    import json
    with open("../../../../config/qgen/config.film.json", 'rb') as f_config:
        config = json.load(f_config)

    QGenNetworkDecoder(config["model"], num_words=354, start_token=1, stop_token=2, policy_gradient=False)
