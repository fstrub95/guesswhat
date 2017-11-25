import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

import neural_toolbox.film_layer as film
import neural_toolbox.ft_utils as ft_utils
import neural_toolbox.rnn as rnn
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


            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='image')
            self.image_out = get_image_features(
                image=self._image, question=self.last_rnn_state,
                is_training=self._is_training,
                scope_name="image_processing",
                config=config['image'],
                dropout_keep=dropout_keep
            )

            assert len(self.image_out.get_shape()) == 4, \
                "Incorrect image input and/or attention mechanism (should be none)"


            #####################
            #   ORALE SIDE INPUT
            #####################



            # Category
            self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

            cat_emb = tfc_layers.embed_sequence(
                ids=self._category,
                vocab_size=config['category']["n_categories"] + 1,
                embed_dim=config['category']["embedding_dim"],
                scope="category_embedding",
                reuse=reuse)


            # Spatial
            self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')

            spatial_emb = tfc_layers.fully_connected(self._spatial,
                                                           num_outputs=config["spatial"]["no_mlp_units"],
                                                           activation_fn=tf.nn.relu,
                                                           reuse=reuse,
                                                           scope="spatial_upsampling")



            # Mask
            self._mask = tf.placeholder(tf.float32, self.image_out.get_shape()[:3], name='mask')
            self._mask = tf.expand_dims(self._mask, axis=-1)

            mask_dim = int(self.image_out.get_shape()[1]) * int(self.image_out.get_shape()[2])


            # Create input embedding
            self.film_input, self.attention_input, self.mlp_input = None, None, None

            self.concat_emb(config, "question", emb=self.last_rnn_state)
            self.concat_emb(config, "category", emb=cat_emb)
            self.concat_emb(config, "spatial", emb=spatial_emb)
            self.concat_emb(config, "mask", emb=tf.reshape(self._mask, shape=[-1, mask_dim]))


            #####################
            #   STEM
            #####################

            with tf.variable_scope("stem", reuse=reuse):

                stem_features = self._image
                if config["stem"]["spatial_location"]:
                    stem_features = ft_utils.append_spatial_location(stem_features)

                if config["stem"]["mask"]:
                    stem_features = tf.concat([stem_features, self._mask], axis=3)


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

                        if config["resblock"]["mask"]:
                            res_output = tf.concat([res_output, self._mask], axis=3)

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

                if config["classifier"]["mask"]:
                    classif_features = tf.concat([classif_features, self._mask], axis=3)

                # 2D-Conv
                self.classif_conv = tfc_layers.conv2d(classif_features,
                                                      num_outputs=config["classifier"]["conv_out"],
                                                      kernel_size=config["classifier"]["conv_kernel"],
                                                      normalizer_fn=tfc_layers.batch_norm,
                                                      normalizer_params={"center": True, "scale": True,
                                                                         "decay": 0.9,
                                                                         "is_training": self._is_training,
                                                                         "reuse": reuse},
                                                      activation_fn=tf.nn.relu,
                                                      reuse=reuse,
                                                      scope="classifier_conv")


                self.pooling = get_attention(self.classif_conv, self.attention_input, config["classifier"]["attention"],
                                             dropout_keep=dropout_keep, reuse=reuse)

                if config["classifier"]["merge"]:

                    self.classif_rnn = tfc_layers.fully_connected(self.mlp_input,
                                                                   num_outputs=config["classifier"]["conv_out"],
                                                                   normalizer_fn=tfc_layers.batch_norm,
                                                                   normalizer_params={"center": True, "scale": True,
                                                                                     "decay": 0.9,
                                                                                     "is_training": self._is_training,
                                                                                     "reuse": reuse},
                                                                   activation_fn=tf.nn.relu,
                                                                   reuse=reuse,
                                                                   scope="classifier_rnn_layer")

                    self.classif_state = self.pooling * self.classif_rnn
                    # self.classif_state = tf.nn.dropout(self.classif_state, dropout_keep)
                else:
                    self.classif_state = tf.concat([self.pooling, self.mlp_input], axis=1)


                self.hidden_state = tfc_layers.fully_connected(self.classif_state,
                                                                   num_outputs=config["classifier"]["no_mlp_units"],
                                                                   normalizer_fn=tfc_layers.batch_norm,
                                                                   normalizer_params={"center": True, "scale": True,
                                                                                      "decay": 0.9,
                                                                                      "is_training": self._is_training,
                                                                                      "reuse": reuse},activation_fn=tf.nn.relu,
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
                self.accuracy = tf.equal(self.prediction, tf.argmax(self._answer, axis=1))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)

            print('Model... build!')

    def concat_emb(self, config, name, emb):
        #TODO factorize code
        if config["film_input"][name]:
            if self.film_input is None:
                self.film_input = emb
            else:
                self.film_input = tf.concat([self.film_input, emb], axis=1)

        if config["attention_input"][name]:
            if self.attention_input is None:
                self.attention_input = emb
            else:
                self.attention_input = tf.concat([self.attention_input, emb], axis=1)

        if config["mlp_input"][name]:
            if self.mlp_input is None:
                self.mlp_input = emb
            else:
                self.mlp_input = tf.concat([self.mlp_input, emb], axis=1)



    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy


if __name__ == "__main__":
    FiLM_Oracle({

    "name" : "CLEVR with FiLM",
    "type" : "film",
    "split_question" : True,

    "film_input":
    {
      "question": True,
      "category": False,
      "spatial": False,
      "mask": True
    },

    "attention_input":
    {
      "question": True,
      "category": False,
      "spatial": False,
      "mask": True
    },

    "mlp_input":
    {
      "question": False,
      "category": True,
      "spatial": True,
      "mask": False
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
      "word_embedding_dim": 100,
      "rnn_state_size": 2048,
      "glove" : True,
      "bidirectional" : True,
      "max_pool" : False
    },

    "category": {
      "n_categories": 90,
      "embedding_dim": 200
    },

    "spatial": {
      "no_mlp_units": 30
    },


    "stem" : {
      "spatial_location" : True,
      "mask": True,
      "conv_out": 256,
      "conv_kernel": [3,3]
    },

    "resblock" : {
      "no_resblock" : 4,
      "spatial_location" : True,
      "mask": True,
      "kernel1" : [1,1],
      "kernel2" : [3,3]
    },

    "classifier" : {
      "spatial_location" : True,
      "mask": True,
      "conv_out": 512,
      "conv_kernel": [1,1],
      "no_mlp_units": 1024,

      "merge" : False,

      "attention" : {
        "mode": "glimpse",
        "no_attention_mlp" : 256,
        "no_glimpses" : 1
      }
    }

  }, no_words=354, no_answers=3)
