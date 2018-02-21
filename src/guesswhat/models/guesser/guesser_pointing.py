import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_utils.abstract_network import AbstractNetwork
from neural_toolbox import rnn
from generic.tf_factory.fusion_factory import get_fusion_mechanism


from generic.tf_factory.image_factory import get_image_features
from neural_toolbox.film_stack import FiLM_Stack
from neural_toolbox import regularizer_toolbox
from neural_toolbox.utils import iou_accuracy

class GuesserNetwork(AbstractNetwork):
    def __init__(self, config, no_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse):

            batch_size = None

            self._is_training = tf.placeholder(tf.bool, name="is_training")

            dropout_keep_scalar = float(config['regularizer'].get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                        lambda: tf.constant(dropout_keep_scalar),
                                        lambda: tf.constant(1.0))

            self.regularizer = regularizer_toolbox.Regularizer(config['regularizer'], self._is_training, dropout_keep, reuse)


            #####################
            #   DIALOGUE
            #####################
            use_glove = config["dialogue"]['glove']
            self._dialogue = tf.placeholder(tf.int32, [batch_size, None], name='dialogue')
            self._seq_length_dialogue = tf.placeholder(tf.int32, [batch_size], name='seq_length_dialogue')

            if use_glove : self._glove = tf.placeholder(tf.float32, [None, None, 300], name="glove")

            with tf.variable_scope('dialogue_embedding'):

                word_emb = tfc_layers.embed_sequence(
                    ids=self._dialogue,
                    vocab_size=no_words,
                    embed_dim=config["dialogue"]["word_embedding_dim"],
                    scope="input_word_embedding",
                    reuse=reuse)

                if use_glove:
                    word_emb = tf.concat([word_emb, self._glove], axis=2)

                # word_emb = tf.nn.dropout(word_emb, dropout_keep)
                with tf.variable_scope('word_embedding_reg'):
                    word_emb = self.regularizer.apply(word_emb)

                # If specified, use a lstm, otherwise default behavior is GRU now
                if config["dialogue"]["use_lstm"] :
                    _, self.dialogue_embedding = rnn.variable_length_LSTM(word_emb,
                                                                          num_hidden=config["dialogue"]['rnn_state_size'],
                                                                          seq_length=self._seq_length_dialogue,
                                                                          dropout_keep_prob=dropout_keep)

                else:
                    _, self.dialogue_embedding = rnn.gru_factory(
                        inputs=word_emb,
                        seq_length=self._seq_length_dialogue,
                        num_hidden=config["dialogue"]["rnn_state_size"],
                        bidirectional=config["dialogue"]["bidirectional"],
                        max_pool=config["dialogue"]["max_pool"],
                        layer_norm=config["dialogue"]["layer_norm"],
                        reuse=reuse)

                #self.dialogue_embedding = tf.nn.dropout(self.dialogue_embedding, dropout_keep)
                with tf.variable_scope('dialogue_reg'):
                    self.dialogue_embedding = self.regularizer.apply(self.dialogue_embedding)

            #####################
            #   IMAGES
            #####################

            if config['image']['use_image']:
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

                #self.image_embedding = tf.nn.dropout(self.image_embedding, dropout_keep)
                with tf.variable_scope("image_embedding_reg"):
                    self.image_embedding = self.regularizer(self.image_embedding)

            else:
                assert config['fusion']['mode'] == "none", "If you don't want to use image, set fusion to none"
                self.image_embedding = None

            #####################
            #   FUSION MECHANISM
            #####################

            if config['dialogue']['reinject_for_fusion']:
                dialog_to_fuse = self.dialogue_embedding
            else:
                assert config['fusion']['mode'] == "none", "If you don't want to reinject dialog, set fusion to none"
                dialog_to_fuse = None


            with tf.variable_scope('fusion'):
                self.visual_dialogue_embedding, _ = get_fusion_mechanism(input1=self.image_embedding,
                                                                      input2=dialog_to_fuse,
                                                                      config=config["fusion"],
                                                                      dropout_keep=dropout_keep,
                                                                      reuse=reuse)

                with tf.variable_scope("fusion_reg"):
                    self.visual_dialogue_embedding = self.regularizer.apply(self.visual_dialogue_embedding)


            #####################
            #   MLP COORD
            #####################

            size_hidden_mlp_coord = config['mlp_coord']['mlp_coord_hidden']
            self.coord_temp = self.visual_dialogue_embedding


            if size_hidden_mlp_coord > 0:

                with tf.variable_scope("mlp_coord_hidden"):
                    self.coord_temp = self.regularizer(self.coord_temp)

                self.coord_temp = tfc_layers.fully_connected(self.coord_temp,
                                                             num_outputs=size_hidden_mlp_coord,
                                                             activation_fn=tf.nn.relu,
                                                             scope='mlp_coord_hidden')

            with tf.variable_scope("mlp_coord_out"):
                self.coord_temp = self.regularizer(self.coord_temp)


            self.coord_out = tfc_layers.fully_connected(self.coord_temp,
                                                     num_outputs=4,
                                                     activation_fn=None,
                                                     scope='mlp_coord_output')

            #####################
            #   LOSS
            #####################

            # Targets
            self._targets = tf.placeholder(tf.float32, [batch_size, 4], name="targets_bbox")

            self.loss = tf.losses.mean_squared_error(labels=self._targets, predictions=self.coord_out)
            self.loss = tf.reduce_mean(self.loss)

            with tf.variable_scope('inter_over_union'):
                self.inter_over_union = iou_accuracy(self._targets, self.coord_out)
                self.inter_over_union = tf.reduce_mean(self.inter_over_union)

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.inter_over_union




if __name__ == "__main__":

    import json
    with open("../../../../config/guesser/config.film.json", 'r') as f_config:
        config = json.load(f_config)

    GuesserNetwork(config["model"], no_words=354)

    print("Done")
