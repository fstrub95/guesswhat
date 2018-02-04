import tensorflow as tf

import tensorflow.contrib.layers as tfc_layers
import tensorflow.contrib.rnn as tfc_rnn
import tensorflow.contrib.seq2seq as tfc_seq

from neural_toolbox import rnn, utils

from generic.tf_factory.image_factory import get_image_features
from generic.tf_factory.fusion_factory import get_fusion_mechanism

from generic.tf_utils.abstract_network import AbstractNetwork

from neural_toolbox.film_stack import FiLM_Stack


from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

import collections

class BasicDecoderWithStateOutput(
       collections.namedtuple('BasicDecoderWithStateOutput', ('rnn_output', 'rnn_state', 'sample_id'))):
   """ Basic Decoder Named Tuple with rnn_output, rnn_state, and sample_id """
pass

class BasicDecoderWithState(tfc_seq.BasicDecoder):

    def __init__(self, cell, helper, initial_state, output_layer=None):
        super(BasicDecoderWithState, self).__init__(cell=cell,
                                                 helper=helper,
                                                 initial_state=initial_state,
                                                 output_layer=output_layer)

    @property
    def output_size(self):
        return BasicDecoderWithStateOutput(rnn_output=self._rnn_output_size(),
                                           rnn_state=tensor_shape.TensorShape([self._cell.output_size]),
                                           sample_id=self._helper.sample_ids_shape)


    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderWithStateOutput(nest.map_structure(lambda _: dtype, self._rnn_output_size()),
                                           dtype,
                                           self._helper.sample_ids_dtype)

    def step(self, time, inputs, state, name=None):

        (outputs, next_state, next_inputs, finished) = super(BasicDecoderWithState, self).step(time, inputs, state, name)

        # store state
        outputs = BasicDecoderWithStateOutput(
            rnn_output=outputs.rnn_output,
            rnn_state=state,
            sample_id=outputs.sample_id)

        return outputs, next_state, next_inputs, finished




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
            self.rnn_states, self.rnn_last_states = rnn.gru_factory(
                inputs=word_emb_dialogue,
                seq_length=self._seq_length_dialogue,
                num_hidden=config["dialogue"]["rnn_state_size"],
                bidirectional=config["dialogue"]["bidirectional"],
                max_pool=config["dialogue"]["max_pool"],
                layer_norm=config["dialogue"]["layer_norm"],
                reuse=reuse)
            self.dialogue_embedding = tf.nn.dropout(self.rnn_last_states, dropout_keep)


            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')
            self.image_out = get_image_features(
                image=self._image,
                question=self.rnn_last_states,
                is_training=self._is_training,
                scope_name="image_processing",
                config=config['image'])


            #####################
            #   FiLM
            #####################

            # Use attention or use vgg features
            if len(self.image_out.get_shape()) == 2:
                self.image_embedding = self.image_out

            else:
                with tf.variable_scope("image_film_stack", reuse=reuse):

                    self.film_img_stack = FiLM_Stack(image=self.image_out,
                                                     film_input=self.dialogue_embedding,
                                                     attention_input=self.dialogue_embedding,
                                                     is_training=self._is_training,
                                                     dropout_keep=dropout_keep,
                                                     config=config["image"]["film_block"],
                                                     reuse=reuse)

                    self.image_embedding =self.film_img_stack.get()
                    self.image_embedding = tf.nn.dropout(self.image_embedding, dropout_keep)


            #####################
            #   FUSION MECHANISM
            #####################

            if config["fusion"]["apply_fusion"]:

                with tf.variable_scope('fusion'):

                    self.final_embedding, need_dropout = get_fusion_mechanism(input1=self.image_embedding,
                                                                  input2=self.dialogue_embedding,
                                                                  config=config["fusion"],
                                                                  dropout_keep=dropout_keep,
                                                                  reuse=reuse)

                    if need_dropout:
                        self.final_embedding = tf.nn.dropout(self.final_embedding, dropout_keep)

            else:
                self.final_embedding = self.image_embedding


            #####################
            #   TARGET QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [batch_size, None], name='question')
            self._seq_length_question = tf.placeholder(tf.int32, [batch_size], name='seq_length_question')

            input_question = self._question[:, :-1]  # Ignore start token
            target_question = self._question[:, 1:]  # Ignore stop token

            self._question_mask = tf.sequence_mask(lengths=self._seq_length_question - 1, dtype=tf.float32) # -1 : remove start token a decoding time

            if config["dialogue"]["share_decoder_emb"]:
                self.question_emb_weights = self.dialogue_emb_weights
            else:
                self.question_emb_weights = tf.get_variable("question_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]])

            self.word_emb_question = tf.nn.embedding_lookup(params=self.question_emb_weights, ids=input_question)
            self.word_emb_question = tf.nn.dropout(self.word_emb_question, dropout_keep)


            #####################
            #   DECODER
            #####################

            self.decoder_cell = rnn.create_cell(num_units=int(self.final_embedding.get_shape()[-1]),
                                                layer_norm=False, # TODO use layer norm if it works!
                                                reuse=reuse)

            self.decoder_projection_layer = utils.MultiLayers(
                [   tf.layers.Dropout(dropout_keep),
                    tf.layers.Dense(num_words, use_bias=False),
                ])

            training_helper = tfc_seq.TrainingHelper(inputs=self.word_emb_question,  # The question is the target
                                                     sequence_length=self._seq_length_question-1) # -1 : remove start token a decoding time

            decoder = BasicDecoderWithState(
                self.decoder_cell, training_helper, self.final_embedding,
                output_layer=self.decoder_projection_layer)

            (self.decoder_outputs, self.decoder_states, _), _ , _ = tfc_seq.dynamic_decode(decoder, maximum_iterations=None)


            #####################
            #   LOSS
            #####################

            # compute the softmax for evaluation
            with tf.variable_scope('ml_loss'):

                self.ml_loss = tfc_seq.sequence_loss(logits=self.decoder_outputs,
                                                  targets=target_question,
                                                  weights=self._question_mask,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True)

                self.softmax_output = tf.nn.softmax(self.decoder_outputs, name="softmax")
                self.argmax_output = tf.argmax(self.decoder_outputs, axis=2)

                self.loss = self.ml_loss

            # Compute policy gradient
            if policy_gradient:

                self.cum_rewards = tf.placeholder(tf.float32, shape=[batch_size, None], name='cum_reward')

                with tf.variable_scope('rl_baseline'):
                    baseline_input = tf.stop_gradient(self.decoder_states)

                    baseline_hidden = tfc_layers.fully_connected(baseline_input,
                                                              num_outputs=int(int(baseline_input.get_shape()[-1])/4),
                                                              activation_fn=tf.nn.relu,
                                                              scope='baseline_hidden',
                                                              reuse=reuse)

                    baseline_hidden = tf.layers.dropout(baseline_hidden, dropout_keep)

                    baseline_out = tfc_layers.fully_connected(baseline_hidden,
                                                              num_outputs=1,
                                                              activation_fn=tf.nn.relu,
                                                              scope='baseline',
                                                              reuse=reuse)
                    baseline_out = tf.squeeze(baseline_out, axis=-1)

                    self.baseline = baseline_out * self._question_mask

                    self.baseline_loss = tf.reduce_sum(tf.square(self.cum_rewards - self.baseline))


                with tf.variable_scope('policy_gradient_loss'):

                    self.log_of_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_outputs, labels=target_question)
                    self.log_of_policy = self.log_of_policy * self._question_mask

                    self.score_function = tf.multiply(self.log_of_policy, self.cum_rewards - self.baseline)

                    self.policy_gradient_loss = tf.reduce_sum(self.score_function, axis=1)  # sum over the dialogue trajectory
                    self.policy_gradient_loss = tf.reduce_mean(self.policy_gradient_loss, axis=0)

                    self.loss = self.policy_gradient_loss



    def create_sampling_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        sample_helper = tfc_seq.SampleEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = tfc_seq.BasicDecoder(
            self.decoder_cell, sample_helper, self.final_embedding,
            output_layer=self.decoder_projection_layer)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        return sample_id, seq_length

    def create_greedy_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._dialogue)[0]
        greedy_helper = tfc_seq.GreedyEmbeddingHelper(embedding=self.question_emb_weights,
                                                      start_tokens=tf.fill([batch_size], start_token),
                                                      end_token=stop_token)

        decoder = tfc_seq.BasicDecoder(
            self.decoder_cell, greedy_helper, self.final_embedding,
            output_layer=self.decoder_projection_layer)

        (_, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

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

    network = QGenNetworkDecoder(config["model"], num_words=354, policy_gradient=True)

    network.create_sampling_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_greedy_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_beam_graph(start_token=1, stop_token=2, max_tokens=10, k_best=5)

