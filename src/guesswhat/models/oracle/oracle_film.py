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
            #   ORACLE SIDE INPUTS
            #####################

            # Category
            if config["inputs"].get("category", True):
                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

                cat_emb = tfc_layers.embed_sequence(
                    ids=self._category,
                    vocab_size=config['category']["n_categories"] + 1,
                    embed_dim=config['category']["embedding_dim"],
                    scope="category_embedding",
                    reuse=reuse)
                cat_emb = tf.nn.dropout(cat_emb, dropout_keep)
            else:
                cat_emb = None

            # Spatial
            if config["inputs"].get("spatial", True):
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                if config["spatial"]["no_mlp_units"] > 0:
                    spatial_emb = tfc_layers.fully_connected(self._spatial,
                                                                   num_outputs=config["spatial"]["no_mlp_units"],
                                                                   activation_fn=tf.nn.relu,
                                                                   reuse=reuse,
                                                                   scope="spatial_upsampling")
                    spatial_emb = tf.nn.dropout(spatial_emb, dropout_keep)
                else:
                    spatial_emb = self._spatial
            else:
                spatial_emb = None

            self.classifier_input = []

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

                # apply attention or use vgg features
                if len(self.image_out.get_shape()) == 2:
                    self.classifier_input.append(self.image_out)

                else:
                    # Compute object mask
                    self._mask = tf.placeholder(tf.float32, self.image_out.get_shape()[:3], name='img_mask')
                    self._mask = tf.expand_dims(self._mask, axis=-1)

                    self.film_img_input = []
                    with tf.variable_scope("image_film_input", reuse=reuse):

                        if config["image"]["film_input"]["question"]:
                            self.film_img_input.append(self.last_rnn_state)

                        if config["image"]["film_input"]["category"]:
                            self.film_img_input.append(cat_emb)

                        if config["image"]["film_input"]["spatial"]:
                            self.film_img_input.append(spatial_emb)

                        if config["image"]["film_input"]["mask"]:
                            mask_dim = int(self.image_out.get_shape()[1]) * int(self.image_out.get_shape()[2])
                            flat_mask = tf.reshape(self._mask, shape=[-1, mask_dim])
                            self.film_crop_input.append(flat_mask)

                        self.film_img_input = tf.concat(self.film_img_input, axis=1)

                    with tf.variable_scope("image_film_stack", reuse=reuse):

                        def append_extra_features(features, config):
                            if config["spatial_location"]: # add the pixel location as two additional feature map
                                features = ft_utils.append_spatial_location(features)
                            if config["mask"]: # add the mask on the object as one additional feature map
                                features = tf.concat([features, self._mask], axis=3)
                            return features


                        self.film_img_stack = FiLM_Stack(image=self.image_out,
                                                         film_input=self.film_img_input,
                                                         attention_input=self.last_rnn_state,
                                                         is_training=self._is_training,
                                                         dropout_keep=dropout_keep,
                                                         config=config["image"]["film_block"],
                                                         append_extra_features=append_extra_features,
                                                         reuse=reuse)

                        self.classifier_input.append(self.film_img_stack.get())

            #####################
            #   CROP
            #####################

            if config["inputs"]["crop"]:
                self._crop = tf.placeholder(tf.float32, [self.batch_size] + config['crop']["dim"], name='crop')
                self.crop_out = get_image_features(
                    image=self._crop, question=self.last_rnn_state,
                    is_training=self._is_training,
                    scope_name="crop_processing",
                    config=config['crop'],
                    dropout_keep=dropout_keep
                )

                # apply attention or use vgg features
                if len(self.crop_out.get_shape()) == 2:
                    self.classifier_input.append(self.crop_out)

                else:

                    self._mask_crop = tf.placeholder(tf.float32, self.crop_out.get_shape()[:3], name='crop_mask')
                    self._mask_crop = tf.expand_dims(self._mask_crop, axis=-1)

                    self.film_crop_input = []
                    with tf.variable_scope("image_film_input", reuse=reuse):

                        if config["crop"]["film_input"]["question"]:
                            self.film_crop_input.append(self.last_rnn_state)

                        if config["crop"]["film_input"]["category"]:
                            self.film_crop_input.append(cat_emb)

                        if config["crop"]["film_input"]["spatial"]:
                            self.film_crop_input.append(spatial_emb)

                        if config["crop"]["film_input"]["mask"]:
                            mask_dim = int(self.crop_out.get_shape()[1]) * int(self.crop_out.get_shape()[2])
                            flat_mask = tf.reshape(self._mask_crop, shape=[-1, mask_dim])
                            self.film_crop_input.append(flat_mask)

                        self.film_crop_input = tf.concat(self.film_crop_input, axis=1)

                    with tf.variable_scope("crop_film_stack", reuse=reuse):

                        def append_extra_features(features, config):
                            if config["spatial_location"]: # add the pixel location as two additional feature map
                                features = ft_utils.append_spatial_location(features)
                            if config["mask"]: # add the mask on the object as one additional feature map
                                features = tf.concat([features, self._mask_crop], axis=3)
                            return features

                        self.film_crop_stack = FiLM_Stack(image=self.crop_out,
                                                          film_input=self.film_crop_input,
                                                          attention_input=self.last_rnn_state,
                                                          is_training=self._is_training,
                                                          dropout_keep=dropout_keep,
                                                          config=config["crop"]["film_block"],
                                                          append_extra_features=append_extra_features,
                                                          reuse=reuse)

                        self.classifier_input.append(self.film_crop_stack.get())


            #####################
            #   FINAL LAYER
            #####################

            with tf.variable_scope("classifier", reuse=reuse):
                if config["classifier"]["inputs"]["question"]:
                    self.classifier_input.append(self.last_rnn_state)

                if config["classifier"]["inputs"]["category"]:
                    self.classifier_input.append(cat_emb)

                if config["classifier"]["inputs"]["spatial"]:
                    self.classifier_input.append(spatial_emb)

                assert len(self.classifier_input) > 0, "Please provide some inputs for the classifier!!!"
                self.classifier_input = tf.concat(self.classifier_input, axis=1)

                self.hidden_state = tfc_layers.fully_connected(self.classifier_input,
                                                               num_outputs=config["classifier"]["no_mlp_units"],
                                                               activation_fn=tf.nn.relu,
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


    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy


if __name__ == "__main__":

    import json
    with open("../../../../config/oracle/config.film.json", 'rb') as f_config:
        config = json.load(f_config)

    FiLM_Oracle(config["model"], no_words=354, no_answers=3)
