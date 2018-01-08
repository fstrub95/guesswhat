import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

import neural_toolbox.ft_utils as ft_utils
import neural_toolbox.rnn as rnn

from generic.tf_factory.image_factory import get_image_features

from generic.tf_utils.abstract_network import ResnetModel
from neural_toolbox.film_stack import FiLM_Stack



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

            word_emb = tf.nn.dropout(word_emb, dropout_keep)
            self.rnn_states, self.last_rnn_state = rnn.gru_factory(
                inputs=word_emb,
                seq_length=self._seq_length,
                num_hidden=config["question"]["rnn_state_size"],
                bidirectional=config["question"]["bidirectional"],
                max_pool=config["question"]["max_pool"],
                reuse=reuse)
            self.last_rnn_state = tf.nn.dropout(self.last_rnn_state, dropout_keep)

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

            self.visual_features = []

            #####################
            #   IMAGES
            #####################


            if config["inputs"]["image"]:
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

                with tf.variable_scope("image_film_stack", reuse=reuse):

                    def append_extra_features(features, config):
                        if config["spatial_location"]:
                            features = ft_utils.append_spatial_location(features)
                        if config["mask"]:
                            features = tf.concat([features, self._mask], axis=3)
                        return features

                    self.film_img_stack = FiLM_Stack(image=self.image_out,
                                                     film_input=self.film_input,
                                                     is_training=self._is_training,
                                                     dropout_keep=dropout_keep,
                                                     config=config["image"],
                                                     append_extra_features=append_extra_features,
                                                     reuse=reuse)

                self.visual_features.append(self.film_img_stack)

            #####################
            #   CROP
            #####################

            if config["inputs"]["crop"]:
                self._crop = tf.placeholder(tf.float32, [self.batch_size] + config['crop']["dim"], name='crop')
                self.crop_out = get_image_features(
                    image=self._image, question=self.last_rnn_state,
                    is_training=self._is_training,
                    scope_name="image_processing",
                    config=config['crop'],
                    dropout_keep=dropout_keep
                )

                assert len(self.crop_out.get_shape()) == 4, \
                    "Incorrect crop input and/or attention mechanism (should be none)"

                with tf.variable_scope("crop_film_stack", reuse=reuse):
                    self.film_crop_stack = FiLM_Stack(image=self.crop_out,
                                                     film_input=self.film_input,
                                                     is_training=self._is_training,
                                                     dropout_keep=dropout_keep,
                                                     config=config["crop"],
                                                     reuse=reuse)

                self.visual_features.append(self.film_crop_stack)


            #####################
            #   FINAL LAYER
            #####################

            assert len(self.visual_features) > 0

            self.classif_state = tf.concat(self.visual_features + [self.mlp_input], axis=1)

            self.hidden_state = tfc_layers.fully_connected(self.classif_state,
                                                               num_outputs=config["final_layer"]["no_mlp_units"],
                                                               reuse=reuse,
                                                               scope="classifier_hidden_layer")

            self.hidden_state = tf.nn.dropout(self.hidden_state, dropout_keep)
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
      "category": True,
      "spatial": False,
      "mask": False,
    },

    "mlp_input":
    {
      "question": False,
      "category": True,
      "spatial": True,
      "mask": False
    },


    "crop":
    {
      "image_input": "raw",
      "dim": [224, 224, 3],
      "normalize": False,

      "attention" : {
        "mode": "none"
      },

      "stem": {
            "spatial_location": True,
            "mask": True,
            "conv_out": 128,
            "conv_kernel": [3, 3]
        },

        "resblock": {
            "no_resblock": 2,
            "spatial_location": True,
            "mask": True,
            "kernel1": [1, 1],
            "kernel2": [3, 3]
        },

        "classifier": {
            "spatial_location": True,
            "mask": False,
            "conv_out": 256,
            "conv_kernel": [1, 1],

            "attention": {
                "mode": "max",
            }
        }

    },

    "image":
    {
      "image_input": "conv",
      "dim": [14, 14, 2048],
      "normalize": False,

      "attention" : {
        "mode": "none"
      },

      "stem": {
            "spatial_location": True,
            "mask": True,
            "conv_out": 256,
            "conv_kernel": [3, 3]
        },

        "resblock": {
            "no_resblock": 2,
            "spatial_location": True,
            "mask": True,
            "kernel1": [1, 1],
            "kernel2": [3, 3]
        },

        "classifier": {
            "spatial_location": True,
            "mask": False,
            "conv_out": 512,
            "conv_kernel": [1, 1],

            "attention": {
                "mode": "max",
            }
        }
    },


    "question": {
      "word_embedding_dim": 300,
      "rnn_state_size": 2048,
      "glove" : False,
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


    "final_layer" :
    {
    "no_mlp_units": 1024,
    }

  }, no_words=354, no_answers=3)
