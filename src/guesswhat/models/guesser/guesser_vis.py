import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_factory.image_factory import get_image_features
from generic.tf_utils.abstract_network import AbstractNetwork

from neural_toolbox import rnn, utils


class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, num_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):

            batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))


            #############
            #   Objects
            #############

            self._obj_mask = tf.placeholder(tf.float32, [batch_size, None], name='obj_mask')
            self._obj_cats = tf.placeholder(tf.int32, [batch_size, None], name='obj_cats')
            self._obj_spats = tf.placeholder(tf.float32, [batch_size, None, config['spat_dim']], name='obj_spats')

            # Targets
            self._targets = tf.placeholder(tf.int32, [batch_size], name="targets_index")

            self.object_cats_emb = tfc_layers.embed_sequence(
                ids=self._obj_cats,
                vocab_size=config["object"]['no_categories'] + 1,
                embed_dim=config["object"]['cat_emb_dim'],
                scope="cat_embedding",
                reuse=reuse)

            self.object_spat_emb = tfc_layers.embed_sequence(
                ids=self._obj_spats,
                vocab_size=8,
                embed_dim=config["object"]['spat_emb_dim'],
                scope="spat_embedding",
                reuse=reuse)

            # TODO : add image + attention

            self.objects_input = tf.concat([self.object_cats_emb, self.object_spat_emb], axis=2)
            self.flat_objects_input = tf.reshape(self.objects_input, shape=[-1, config["object"]['cat_emb_dim'] + config['spat_dim']])

            with tf.variable_scope('obj_mlp'):
                h1 = tfc_layers.fully_connected(
                    self.flat_objects_input,
                    num_outputs=config["object"]['obj_mlp_units'],
                    activation=tf.nn.relu,
                    scope='l1',
                    reuse=reuse)

                h2 = tfc_layers.fully_connected(
                    h1,
                    num_outputs=config["object"]['dialog_emb_dim'],
                    activation=tf.nn.relu,
                    scope='l2',
                    reuse=reuse)

            self.object_embeddings = tf.reshape(h2, shape=[-1, tf.shape(self._obj_cats)[1], config['dialog_emb_dim']])


            #############
            #   DIALOGUE
            #############

            # Dialogues
            self._dialogues = tf.placeholder(tf.int32, [batch_size, None], name='dialogues')
            self._seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')

            word_emb = tfc_layers.embed_sequence(
                ids=self._dialogues,
                vocab_size=num_words,
                embed_dim=config["word_emb_dim"],
                scope="input_word_embedding",
                reuse=reuse)

            self.rnn_states, self.last_rnn_state = rnn.gru_factory(
                inputs=word_emb,
                seq_length=self._seq_length,
                num_hidden=config["question"]["rnn_state_size"],
                bidirectional=config["question"]["bidirectional"],
                max_pool=config["question"]["max_pool"],
                reuse=reuse)

            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')
            self.image_out = get_image_features(
                image=self._image, question=self.last_rnn_state,
                is_training=self._is_training,
                scope_name="image_processing",
                config=config['image'])


            #####################
            #   VIS
            #####################
            activation_name = config["activation"]
            with tf.variable_scope('final_mlp'):

                self.dialogue_projection = utils.fully_connected(self.last_rnn_state, config["no_question_mlp"], activation=activation_name, scope='question_mlp')
                self.image_projection = utils.fully_connected(self.image_out, config["no_image_mlp"], activation=activation_name, scope='image_mlp')

                full_embedding = self.image_projection * self.dialogue_projection
                full_embedding = tf.nn.dropout(full_embedding, dropout_keep)

                self.full_projection = utils.fully_connected(full_embedding, config["no_hidden_final_mlp"], scope='layer1', activation=activation_name)
                self.full_projection = tf.nn.dropout(self.full_projection, dropout_keep)


            #####################
            #   SOFTMAX
            #####################

            scores = tf.matmul(self.object_embeddings, self.full_projection)
            scores = tf.reshape(scores, [-1, tf.shape(self._obj_cats)[1]])

            self.softmax = utils.masked_softmax(scores, self._obj_mask)
            self.selected_object = tf.argmax(self.softmax, axis=1)

            self.loss = tf.reduce_mean(utils.cross_entropy(self.softmax, self._targets))

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.selected_object, self._targets)
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))


    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy

