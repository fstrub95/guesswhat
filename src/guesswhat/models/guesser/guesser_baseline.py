import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_utils.abstract_network import AbstractNetwork
from neural_toolbox import rnn, utils
import neural_toolbox.ft_utils as ft_utils

from generic.tf_factory.image_factory import get_image_features
from neural_toolbox.film_stack import FiLM_Stack


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

            use_image = config.get("image", False)
            if use_image:
                use_film = "film_input" in config['image']
            else:
                use_film = False

            # In config file, if the size of the projection is not specified for dialogue, don't project it at all
            # The object embedding will be projected to match the lstm output
            if config['dialog_emb_dim'] != 0:
                project_vizdial_embedding = True
                dialog_emb_dim = config['dialog_emb_dim']
            elif use_film :
                project_vizdial_embedding = False
                dialog_emb_dim = 1024  # Dim out of film and attention, Ugly
            else:
                project_vizdial_embedding = False
                dialog_emb_dim = config["rnn_config"]['num_rnn_units']

            # Dialogues
            self._dialogues = tf.placeholder(tf.int32, [batch_size, None], name='dialogues')
            self._seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')

            # Objects
            #self._obj_mask = tf.placeholder(tf.float32, [batch_size, None], name='obj_mask')
            self._num_object = tf.placeholder(tf.int32, [batch_size], name='num_obj')
            self._obj_cats = tf.placeholder(tf.int32, [batch_size, None], name='obj_cats')
            self._obj_spats = tf.placeholder(tf.float32, [batch_size, None, config['spat_dim']], name='obj_spats')

            # Targets
            self._targets = tf.placeholder(tf.int32, [batch_size], name="targets_index")

            # Embedding object categories
            self.object_cats_emb = tfc_layers.embed_sequence(
                ids=self._obj_cats,
                vocab_size=config['no_categories'] + 1,
                embed_dim=config['cat_emb_dim'],
                scope="cat_embedding",
                reuse=reuse)

            # Adding spatial coordinate (should be optionnal)
            self.objects_input = tf.concat([self.object_cats_emb, self._obj_spats], axis=2)
            self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])

            with tf.variable_scope('obj_mlp'):

                h1 = tfc_layers.fully_connected(self.flat_objects_inp,
                                                num_outputs=config['obj_emb_hidden'],
                                                activation_fn=tf.nn.relu,
                                                scope='l1')

                # h1 = utils.fully_connected(         #replace with tfc
                #     self.flat_objects_inp,
                #     n_out=config['obj_emb_hidden'],
                #     activation='relu',
                #     scope='l1')

                # h1 = tf.nn.dropout(h1, dropout_keep_scalar)

                h2 = tfc_layers.fully_connected(h1,
                                                num_outputs=dialog_emb_dim,
                                                activation_fn=tf.nn.relu,
                                                scope='l2')
                # h2 = utils.fully_connected(
                #     h1,
                #     n_out=dialog_emb_dim,
                #     activation='relu',
                #     scope='l2')

                # h2 = tf.nn.dropout(h2, dropout_keep_scalar)

            obj_embs = tf.reshape(h2, [-1, tf.shape(self._obj_cats)[1], dialog_emb_dim])

            # Compute the word embedding
            word_emb = tfc_layers.embed_sequence(
                ids=self._dialogues,
                vocab_size=num_words,
                embed_dim=config["word_emb_dim"],
                scope="input_word_embedding",
                reuse=reuse)

            # If specified, use a lstm, otherwise default behavior is GRU now
            if config["rnn_config"].get("use_lstm", False):
                _, self.visual_dialogue_embedding = rnn.variable_length_LSTM(word_emb,
                                                          num_hidden=config["rnn_config"]['num_rnn_units'],
                                                          seq_length=self._seq_length,
                                                          dropout_keep_prob=dropout_keep)

            else:
                _, self.visual_dialogue_embedding = rnn.gru_factory(
                    inputs=word_emb,
                    seq_length=self._seq_length,
                    num_hidden=config["rnn_config"]["num_rnn_units"],
                    bidirectional=config["rnn_config"]["bidirectional"],
                    max_pool=config["rnn_config"]["max_pool"],
                    reuse=reuse)

            #####################
            #   IMAGES
            #####################
            if use_image:

                self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')
                self.image_out = get_image_features(
                    image=self._image, question=self.visual_dialogue_embedding,
                    is_training=self._is_training,
                    scope_name="image_processing",
                    config=config['image'],
                    dropout_keep=dropout_keep)

                if use_film:
                    self.film_img_input = []
                    with tf.variable_scope("image_film_input", reuse=reuse):
                        if config["image"]["film_input"]["question"]:
                            self.film_img_input.append(self.visual_dialogue_embedding)
                        else:
                            raise NotImplementedError("Can only use dialog to condition image at the moment")

                        self.film_img_input = tf.concat(self.film_img_input, axis=1)

                    with tf.variable_scope("image_film_stack", reuse=reuse):

                        def append_extra_features(features, config):
                            if config["spatial_location"]:  # add the pixel location as two additional feature map
                                features = ft_utils.append_spatial_location(features)
                            return features

                        self.film_img_stack = FiLM_Stack(image=self.image_out,
                                                         film_input=self.film_img_input,
                                                         attention_input=self.visual_dialogue_embedding,
                                                         is_training=self._is_training,
                                                         dropout_keep=dropout_keep,
                                                         config=config["image"]["film_block"],
                                                         append_extra_features=append_extra_features,
                                                         reuse=reuse)

                        self.visual_dialogue_embedding = self.film_img_stack.get()
                        # TODO dropout


                # If film not used and attention , concatenate dialogue embedding and image features
                elif config["image"]["attention"].get("reinject_dial", True):
                    self.visual_dialogue_embedding = tf.concat([self.visual_dialogue_embedding, self.image_out], axis=-1)


            if project_vizdial_embedding:
                self.visual_dialogue_projection = utils.fully_connected(
                    self.visual_dialogue_embedding,
                    n_out=dialog_emb_dim,
                    activation='relu',
                    scope='visual_dialogue_projection')
                    #TODO dropout

            else:
                self.visual_dialogue_projection = self.visual_dialogue_embedding

            # Compute vector product product
            self.visual_dialogue_projection = tf.expand_dims(self.visual_dialogue_projection, axis=-1)

            scores = tf.matmul(obj_embs, self.visual_dialogue_projection)
            self.scores = tf.reshape(scores, [-1, tf.shape(self._obj_cats)[1]])

            score_mask = tf.sequence_mask(self._num_object)

            score_mask_values = float("-inf") * tf.ones_like(self.scores)

            self.score_masked = tf.where(score_mask, self.scores, score_mask_values)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._targets, logits=self.score_masked)
            self.loss = tf.reduce_mean(self.loss)

            # def masked_softmax(scores, mask):
            #     # subtract max for stability
            #     scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keep_dims=True), [1, tf.shape(scores)[1]])
            #     # compute padded softmax
            #     exp_scores = tf.exp(scores)
            #     exp_scores *= mask
            #     exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keep_dims=True)
            #     return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])
            #
            # self.softmax = masked_softmax(scores, self._obj_mask)

            self.selected_object = tf.argmax(self.score_masked, axis=1)

            with tf.variable_scope('accuracy'):
                self.accuracy = tf.equal(self.selected_object, tf.cast(self._targets,  tf.int64))
                self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy




if __name__ == "__main__":

    GuesserNetwork({
    "word_emb_dim": 300,
    "num_rnn_units": 1024,
    "cat_emb_dim": 256,
    "obj_emb_hidden": 512,

    "dialog_emb_dim": 1024,

    "spat_dim": 8,
    "no_categories": 90,

    "use_image" : True,
    "image":
    {
      "image_input": "conv",
      "dim": [14, 14, 2048],
      "normalize": False,

      "attention" : {
        "mode": "classic",
        "no_attention_mlp" : 256,
        "fuse_mode" : "concat"
      }

    },
        "dropout_keep_prob": 0.5
    }, num_words=78)