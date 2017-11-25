import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

import neural_toolbox.film_layer as film
import neural_toolbox.ft_utils as ft_utils
import neural_toolbox.rnn as rnn
import neural_toolbox.utils
from generic.tf_factory.attention_factory import get_attention
from generic.tf_factory.image_factory import get_image_features
from generic.tf_utils.abstract_network import ResnetModel



class FiLM_Oracle(ResnetModel):
    def __init__(self, config, no_words, no_answers, reuse=False, device=''):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):

            self.batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))

            #####################
            #   QUESTION
            #####################

            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
            self._answer = tf.placeholder(tf.int64, [self.batch_size, no_answers], name='answer')

            word_emb = tfc_layers.embed_sequence(
                ids=self._question,
                vocab_size=no_words,
                embed_dim=config["question"]["word_embedding_dim"],
                scope="word_embedding",
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

            self._image = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='image')
            self.image_out = get_image_features(
                image=self._image, question=self.last_rnn_state,
                is_training=self._is_training,
                scope_name="image_processing",
                config=config['image']
            )

            assert len(self._image.get_shape()) == 4, \
                "Incorrect image input and/or attention mechanism (should be none)"


            #####################
            #   ORALE SIDE INPUT
            #####################

            film_input = self.last_rnn_state

            if config['inputs']['category']:
                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

                cat_emb = tfc_layers.embed_sequence(
                    ids=self._category,
                    vocab_size=config['category']["n_categories"],
                    embed_dim=config['category']["embedding_dim"],
                    scope="category_embedding",
                    reuse=reuse)

                film_input = tf.concat([film_input, cat_emb], axis=1)


            if config['inputs']['spatial']:
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')

                spatial_emb = tfc_layers.fully_connected(self._spatial,
                                                               num_outputs=config["spatial"]["no_mlp_units"],
                                                               activation_fn=tf.nn.relu,
                                                               reuse=reuse,
                                                               scope="spatial_upsampling")

                film_input = tf.concat([film_input, spatial_emb], axis=1)

            if config["inputs"]["mask"]:
                self._mask = tf.placeholder(tf.float32, self.image_out.get_shape()[:3], name='mask')
                self._mask = tf.expand_dims(self._mask, axis=-1)

            #####################
            #   STEM
            #####################

            with tf.variable_scope("stem", reuse=reuse):

                stem_features = self._image
                if config["stem"]["spatial_location"]:
                    stem_features = ft_utils.append_spatial_location(stem_features)

                if config["inputs"]["mask"]:
                    stem_features = tf.concat([stem_features, self._mask], axis=3)


                self.stem_conv = tfc_layers.conv2d(stem_features,
                                                   num_outputs=config["stem"]["conv_out"],
                                                   kernel_size=config["stem"]["conv_kernel"],
                                                   normalizer_fn=tf.layers.batch_normalization,
                                                   normalizer_params={"training": self._is_training, "reuse": reuse},
                                                   activation_fn=tf.nn.relu,
                                                   reuse=reuse,
                                                   scope="stem_conv")

            #####################
            #   FiLM Layers
            #####################

            self.unit_state = tf.random_uniform(shape=self.last_rnn_state.get_shape(),minval=-1, maxval=1, dtype=tf.float32)

            with tf.variable_scope("resblocks", reuse=reuse):

                res_output = self.stem_conv
                self.resblocks = []

                for i in range(config["resblock"]["no_resblock"]):
                    with tf.variable_scope("ResBlock_{}".format(i), reuse=reuse):

                        if config["inputs"]["mask"]:
                            res_output = tf.concat([res_output, self._mask], axis=3)

                        neural_toolbox.

                        resblock = film.FiLMResblock(res_output, film_input,
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
                                                      num_outputs=config["classifier"]["conv_out"],
                                                      kernel_size=config["classifier"]["conv_kernel"],
                                                      normalizer_fn=tf.layers.batch_normalization,
                                                      normalizer_params={"training": self._is_training, "reuse": reuse},
                                                      activation_fn=tf.nn.relu,
                                                      reuse=reuse,
                                                      scope="classifier_conv")


                self.pooling = get_attention(self.classif_conv, film_input, config["classifier"]["attention"],
                                             dropout_keep=dropout_keep, reuse=reuse)

                if config["classifier"]["merge"]:

                    # contextual_input = tf.nn.dropout(film_input, dropout_keep)

                    self.classif_rnn = tfc_layers.fully_connected(film_input,
                                                                   num_outputs=config["classifier"]["conv_out"],
                                                                   normalizer_fn=tf.layers.batch_normalization,
                                                                   normalizer_params= {"training": self._is_training, "reuse": reuse},
                                                                   activation_fn=tf.nn.relu,
                                                                   reuse=reuse,
                                                                   scope="classifier_rnn_layer")

                    self.classif_state = self.pooling * self.classif_rnn
                    # self.classif_state = tf.nn.dropout(self.classif_state, dropout_keep)
                else:
                    self.classif_state = self.pooling


                self.hidden_state = tfc_layers.fully_connected(self.classif_state,
                                                                   num_outputs=config["classifier"]["no_mlp_units"],
                                                                   normalizer_fn=tf.layers.batch_normalization,
                                                                   normalizer_params= {"training": self._is_training, "reuse": reuse},
                                                                   activation_fn=tf.nn.relu,
                                                                   reuse=reuse,
                                                                   scope="classifier_hidden_layer")

                self.out = tfc_layers.fully_connected(self.hidden_state,
                                                             num_outputs=no_answers,
                                                             activation_fn=None,
                                                             reuse=reuse,
                                                             scope="classifier_softmax_layer")

            #####################
            #   Loss
            #####################

            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self._answer, name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy)

            self.softmax = tf.nn.softmax(self.out, name='answer_prob')
            self.prediction = tf.argmax(self.out, axis=1, name='predicted_answer')  # no need to compute the softmax

            self.success = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))  # no need to compute the softmax


            with tf.variable_scope('accuracy'):
                self.accuracy = 1 - tf.reduce_mean(utils.error(self.out, self._answer))

            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy


if __name__ == "__main__":
    FiLM_Oracle({


    "name" : "CLEVR with FiLM",
    "type" : "film",

    "inputs":
    {
      "category": True,
      "spatial": True,
      "mask": True
    },

    "image":
    {
      "image_input": "conv",
      "dim": [14, 14, 2048],
      "normalize": False,

      "attention" : {
        "mode": "none"
      }

    },

    "question": {
      "word_embedding_dim": 200,
      "rnn_state_size": 2048,
      "bidirectional" : True,
      "layer_norm" : False,
      "attention" : True,
      "max_pool" : False
    },

    "category": {
      "n_categories": 90,
      "embedding_dim": 200
    },

    "spatial": {
      "no_mlp_units": 200
    },


    "stem" : {
      "spatial_location" : True,
      "conv_out": 128,
      "conv_kernel": [3,3]
    },

    "resblock" : {
      "no_resblock" : 6,
      "spatial_location" : True,
      "kernel1" : [1,1],
      "kernel2" : [3,3]
    },

    "classifier" : {
      "spatial_location" : True,
      "conv_out": 512,
      "conv_kernel": [1,1],
      "no_mlp_units": 1024
    }

  }, no_words=354, no_answers=3)
