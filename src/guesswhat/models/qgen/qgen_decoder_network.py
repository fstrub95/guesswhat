import tensorflow as tf

import tensorflow.contrib.layers as tfc_layers
import tensorflow.contrib.rnn as tfc_rnn
import tensorflow.contrib.seq2seq as tfc_seq

from generic.tf_factory.image_factory import get_image_features
from generic.tf_utils.abstract_network import AbstractNetwork

import neural_toolbox.rnn as rnn


class QGenNetworkDecoder(AbstractNetwork):

    #TODO: add dropout
    def __init__(self, config, num_words, policy_gradient, device='', reuse=False):
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
                question=None,  # no attention at this point
                is_training=self._is_training,
                scope_name="image_processing",
                config=config['image'])

            # TODO remove after FiLM
            self.image_out = tf.nn.dropout(self.image_out, dropout_keep)

            #####################
            #   DIALOGUE
            #####################

            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length_dialogue = tf.placeholder(tf.int32, [batch_size], name='seq_length_dialogue')

            with tf.variable_scope('word_embedding', reuse=reuse):
                self.dialogue_emb_weights = tf.get_variable("dialogue_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]])

            word_emb_dialogue = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=self._dialogue)

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
            self._question_mask = tf.placeholder(tf.float32, [batch_size, None], name='question_mask')

            if config["dialogue"]["share_decoder_emb"]:
                self.question_emb_weights = self.dialogue_emb_weights
            else:
                self.question_emb_weights = tf.get_variable("question_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]])

            self.word_emb_question = tf.nn.embedding_lookup(params=self.question_emb_weights, ids=self._question)
            self.word_emb_question = tf.nn.dropout(self.word_emb_question, dropout_keep)

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

            self.decoder_cell = tfc_rnn.GRUCell(num_units=config["top"]["decoder"]["num_units"],
                                                activation=tf.nn.tanh,
                                                reuse=reuse)

            self.decoder_projection_layer = tf.layers.Dense(num_words, use_bias=False)

            training_helper = tfc_seq.TrainingHelper(inputs=self.word_emb_question,  # The question is the target
                                                     sequence_length=self._seq_length_question)

            (decoder_outputs, _), _ , _ = self._create_decoder_graph(training_helper, max_tokens=None)

            #####################
            #   LOSS
            #####################

            # compute the softmax for evaluation
            with tf.variable_scope('decoder_output'):
                self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_outputs, labels=self._question)
                self.cross_entropy = self.cross_entropy * self._question_mask

                count = tf.reduce_sum(self._question_mask)

                self.cross_entropy_loss = tf.reduce_sum(self.cross_entropy)
                self.cross_entropy_loss = self.cross_entropy_loss / count

                self.softmax_output = tf.nn.softmax(decoder_outputs, name="softmax")
                self.argmax_output = tf.argmax(decoder_outputs, axis=2)

                self.ml_loss = self.cross_entropy_loss


                if not policy_gradient:
                    self.loss = self.cross_entropy_loss

            # TEMPO : DO NOT USE THOSE LOSS -> TODO imolement RL loss
            self.loss = self.cross_entropy_loss
            self.policy_gradient_loss = self.cross_entropy_loss
            self.baseline_loss = self.cross_entropy_loss


    def _create_decoder_graph(self, helper, max_tokens):

        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, helper, self.final_embedding,
            output_layer=self.decoder_projection_layer)

        return tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

    def create_sampling_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        sample_helper = tfc_seq.SampleEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        (_, sample_id), _, seq_length = self._create_decoder_graph(sample_helper, max_tokens=max_tokens)

        return sample_id, seq_length

    def create_greedy_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        greedy_helper = tfc_seq.GreedyEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        (_, sample_id), _, seq_length = self._create_decoder_graph(greedy_helper, max_tokens=max_tokens)

        return sample_id, seq_length

    def create_beam_graph(self, start_token, stop_token, max_tokens, k_best):

        # create k_beams
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
            self.final_embedding, multiplier=k_best)

        # Define a beam-search decoder
        batch_size = tf.shape(self._dialogue)[0]
        decoder = tfc_seq.BeamSearchDecoder(
            cell=self.decoder_cell,
            embedding=self.question_emb_weights,
            start_tokens=tf.fill([batch_size], start_token),
            end_token=stop_token,
            initial_state=decoder_initial_state,
            beam_width=k_best,
            output_layer=self.decoder_projection_layer,
            length_penalty_weight=0.0)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        return sample_id, seq_length

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.loss


if __name__ == "__main__":

    import json
    with open("../../../../config/qgen/config.film.json", 'rb') as f_config:
        config = json.load(f_config)

    network = QGenNetworkDecoder(config["model"], num_words=354, policy_gradient=False)

    network.create_sampling_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_greedy_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_beam_graph(start_token=1, stop_token=2, max_tokens=10, k_best=5)

