import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers

from generic.tf_utils.abstract_network import AbstractNetwork
from neural_toolbox import rnn
import neural_toolbox.ft_utils as ft_utils

from generic.tf_factory.image_factory import get_image_features
from neural_toolbox.film_stack import FiLM_Stack


