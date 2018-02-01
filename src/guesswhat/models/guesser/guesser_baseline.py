import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_utils.abstract_network import AbstractNetwork
from neural_toolbox import rnn
from generic.tf_factory.fusion_factory import get_fusion_mechanism

from generic.tf_factory.image_factory import get_image_features
from neural_toolbox.film_stack import FiLM_Stack


class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, no_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):

            batch_size = None
            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))



            #####################
            #   DIALOGUE
            #####################


            # Objects
            #self._obj_mask = tf.placeholder(tf.float32, [batch_size, None], name='obj_mask')
            self._num_object = tf.placeholder(tf.int32, [batch_size], name='obj_seq_length')
            self._obj_cats = tf.placeholder(tf.int32, [batch_size, None], name='obj_cats')
            self._obj_spats = tf.placeholder(tf.float32, [batch_size, None, config["object"]['spat_dim']], name='obj_spats')

            # Embedding object categories
            with tf.variable_scope('object_embedding'):

                    self.object_cats_emb = tfc_layers.embed_sequence(
                        ids=self._obj_cats,
                        vocab_size=config["object"]['no_categories'] + 1,
                        embed_dim=config["object"]['cat_emb_dim'],
                        scope="cat_embedding",
                        reuse=reuse)

                    # Adding spatial coordinate (should be optionnal)
                    self.objects_input = tf.concat([self.object_cats_emb, self._obj_spats], axis=2)


                    object_emb_hidden = tfc_layers.fully_connected(self.objects_input,
                                                    num_outputs=config["object"]['obj_emb_hidden'],
                                                    activation_fn=tf.nn.relu,
                                                    scope='obj_mlp_hidden_layer')
                    object_emb_hidden = tf.nn.dropout(object_emb_hidden, dropout_keep_scalar)

                    self.object_embedding = tfc_layers.fully_connected(object_emb_hidden,
                                                    num_outputs=config["object"]['obj_emb_dim'],
                                                    activation_fn=tf.nn.relu,
                                                    scope='obj_mlp_out')

            #####################
            #   DIALOGUE
            #####################

            # Dialogues
            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length_dialogue = tf.placeholder(tf.int32, [batch_size], name='seq_length_dialogue')

            with tf.variable_scope('dialogue_embedding'):

                word_emb = tfc_layers.embed_sequence(
                    ids=self._dialogue,
                    vocab_size=no_words,
                    embed_dim=config["dialogue"]["word_embedding_dim"],
                    scope="input_word_embedding",
                    reuse=reuse)

                if config["dialogue"]['glove']:
                    self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")
                    word_emb = tf.concat([word_emb, self._glove], axis=2)

                word_emb = tf.nn.dropout(word_emb, dropout_keep)

                _, self.dialogue_embedding = rnn.gru_factory(
                    inputs=word_emb,
                    seq_length=self._seq_length_dialogue,
                    num_hidden=config["dialogue"]["rnn_state_size"],
                    bidirectional=config["dialogue"]["bidirectional"],
                    max_pool=config["dialogue"]["max_pool"],
                    reuse=reuse)

                self.dialogue_embedding = tf.nn.dropout(self.dialogue_embedding, dropout_keep)

            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')
            self.image_out = get_image_features(
                image=self._image,
                question=self.dialogue_embedding,
                is_training=self._is_training,
                scope_name="image_processing",
                config=config['image'],
                reuse=reuse)

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

                    self.image_embedding = self.film_img_stack.get()

            self.image_embedding = tf.nn.dropout(self.image_embedding, dropout_keep)

            #####################
            #   FUSION MECHANISM
            #####################

            if config["fusion"]["apply_fusion"]:

                with tf.variable_scope('fusion'):
                    self.visual_dialogue_embedding, _ = get_fusion_mechanism(input1=self.image_embedding,
                                                                          input2=self.dialogue_embedding,
                                                                          config=config["fusion"],
                                                                          dropout_keep=dropout_keep,
                                                                          reuse=reuse)

                    # Note: do not apply dropout here (special case because of scalar product)
            else:
                self.visual_dialogue_embedding = self.image_embedding



            if config["fusion"]["visual_dialogue_projection"] > 0:

                self.visual_dialogue_embedding = tfc_layers.fully_connected(self.visual_dialogue_embedding,
                                           num_outputs=config["fusion"]["visual_dialogue_projection"],
                                           activation_fn=tf.nn.relu,
                                           scope='visual_dialogue_projection')


            #####################
            #   SCALAR PRODUCT
            #####################

            with tf.variable_scope('scalar_product', reuse=reuse):

                # Compute vector product product
                self.visual_dialogue_embedding = tf.expand_dims(self.visual_dialogue_embedding, axis=1)

                self.object_dialogue_matching = self.visual_dialogue_embedding * self.object_embedding

                self.object_dialogue_matching = tf.nn.dropout(self.object_dialogue_matching, keep_prob=dropout_keep)
                self.scores = tf.reduce_sum(self.object_dialogue_matching, axis=2)

            #####################
            #   OBJECT MASKING
            #####################

            with tf.variable_scope('object_mask', reuse=reuse):

                object_mask = tf.sequence_mask(self._num_object)
                score_mask_values = float("-inf") * tf.ones_like(self.scores)

                self.score_masked = tf.where(object_mask, self.scores, score_mask_values)

            #####################
            #   LOSS
            #####################

            # Targets
            self._targets = tf.placeholder(tf.int32, [batch_size], name="targets_index")

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._targets, logits=self.score_masked)
            self.loss = tf.reduce_mean(self.loss)

            self.selected_object = tf.argmax(self.score_masked, axis=1)

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.selected_object, tf.cast(self._targets,  tf.int64))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy




if __name__ == "__main__":

    import json
    with open("../../../../config/guesser/config.film.json", 'r') as f_config:
        config = json.load(f_config)

    GuesserNetwork(config["model"], no_words=354)

    print("Done")
