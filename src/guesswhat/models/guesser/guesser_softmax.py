import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_utils.abstract_network import AbstractNetwork
from generic.tf_factory.attention_factory import get_attention

from neural_toolbox import rnn, utils, attention
import neural_toolbox.film_layer as film
import neural_toolbox.ft_utils as ft_utils


class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, no_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        mini_batch_size = None

        with tf.variable_scope(self.scope_name, reuse=reuse):

            self.batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))



            #############
            #   Objects
            #############

            self.obj_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='obj_mask')
            self.obj_cats = tf.placeholder(tf.int32, [mini_batch_size, None], name='obj_cats')
            self.obj_spats = tf.placeholder(tf.float32, [mini_batch_size, None, config['spat_dim']], name='obj_spats')

            # Targets
            self.targets = tf.placeholder(tf.int32, [mini_batch_size], name="targets_index")


            self.object_cats_emb = tfc_layers.embed_sequence(
                ids=self.obj_cats,
                vocab_size=config["object"]['no_categories'] + 1,
                embed_dim=config["object"]['cat_emb_dim'],
                scope="cat_embedding",
                reuse=reuse)

            self.object_spat_emb = tfc_layers.embed_sequence(
                ids=self.obj_spats,
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

            self.object_embeddings = tf.reshape(h2, shape=[-1, tf.shape(self.obj_cats)[1], config['dialog_emb_dim']])


            #############
            #   Visual Dialogue
            #############

            # Dialogues
            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='dialogues')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=no_words,
                embed_dim=config["question"]["word_embedding_dim"],
                scope="word_embedding",
                reuse=reuse)

            if config["question"]['glove']:
                self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                word_emb = tf.concat([word_emb, self._glove], axis=2)

            self.rnn_states, self.last_rnn_state = rnn.gru_factory(
                inputs=word_emb,
                seq_length=self._seq_length,
                num_hidden=config["question"]["rnn_state_size"],
                bidirectional=config["question"]["bidirectional"],
                max_pool=config["question"]["max_pool"],
                reuse=reuse)


            self.film_input = self.last_rnn_state

            #####################
            #   STEM
            #####################

            with tf.variable_scope("stem", reuse=reuse):

                stem_features = self._image
                if config["stem"]["spatial_location"]:
                    stem_features = ft_utils.append_spatial_location(stem_features)

                self.stem_conv = tfc_layers.conv2d(stem_features,
                                                   num_outputs=config["stem"]["conv_out"],
                                                   kernel_size=config["stem"]["conv_kernel"],
                                                   normalizer_fn=tfc_layers.batch_norm,
                                                   normalizer_params={"center": True, "scale": True,
                                                                      "decay": 0.9,
                                                                      "is_training": self._is_training,
                                                                      "reuse": reuse},
                                                   activation_fn=tf.nn.relu,
                                                   reuse=reuse,
                                                   scope="stem_conv")

            #####################
            #   FiLM Layers
            #####################

            with tf.variable_scope("resblocks", reuse=reuse):

                res_output = self.stem_conv
                self.resblocks = []

                for i in range(config["resblock"]["no_resblock"]):
                    with tf.variable_scope("ResBlock_{}".format(i), reuse=reuse):

                        resblock = film.FiLMResblock(res_output, self.film_input,
                                                     kernel1=config["resblock"]["kernel1"],
                                                     kernel2=config["resblock"]["kernel2"],
                                                     spatial_location=config["resblock"]["spatial_location"],
                                                     is_training=self._is_training,
                                                     reuse=reuse)

                        self.resblocks.append(resblock)
                        res_output = resblock.get()


            #####################
            #   Classifier
            #####################

            with tf.variable_scope("classifier", reuse=reuse):

                classif_features = res_output
                if config["classifier"]["spatial_location"]:
                    classif_features = ft_utils.append_spatial_location(classif_features)

                # 2D-Conv
                self.classif_conv = tfc_layers.conv2d(classif_features,
                                                      num_outputs=config['dialog_emb_dim'],
                                                      kernel_size=config["classifier"]["conv_kernel"],
                                                      normalizer_fn=tfc_layers.batch_norm,
                                                      normalizer_params={"center": True, "scale": True,
                                                                         "decay": 0.9,
                                                                         "is_training": self._is_training,
                                                                         "reuse": reuse},
                                                      activation_fn=tf.nn.relu,
                                                      reuse=reuse,
                                                      scope="classifier_conv")

                self.visual_dialogue = get_attention(self.classif_conv, self.last_rnn_state, config["classifier"]["attention"],
                                             dropout_keep=dropout_keep, reuse=reuse)


            #####################
            #   Sofmax dot-product
            #####################

            scores = tf.matmul(self.object_embeddings, self.visual_dialogue)
            scores = tf.reshape(scores, [-1, tf.shape(self.obj_cats)[1]])

            self.softmax = utils.masked_softmax(scores, self.obj_mask)
            self.selected_object = tf.argmax(self.softmax, axis=1)

            self.loss = tf.reduce_mean(utils.cross_entropy(self.softmax, self.targets))

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.selected_object, self.targets)
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))


    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy



