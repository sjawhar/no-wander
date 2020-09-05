import logging
import numpy as np
import os
import tensorflow as tf

os.environ["TF_KERAS"] = "1"
from keras_multi_head import MultiHeadAttention
from keras_pos_embd import TrigPosEmbedding
from keras_position_wise_feed_forward import FeedForward
from tensorflow.keras.layers import Add, Dropout, LayerNormalization

logger = logging.getLogger(__name__)


def wrap_residual_with_dropout(
    input_layer, name, NextLayer, dropout, epsilon, **kwargs
):
    logger.debug(f'Adding layer "{name}" - {NextLayer.__name__} w/ residual: {kwargs}')
    next_layer = NextLayer(name=name, **kwargs)(input_layer)
    if dropout:
        next_layer = Dropout(rate=dropout, name=f"{name}_dropout")(next_layer)
    residual_layer = Add(name=f"{name}_res")([input_layer, next_layer])
    return LayerNormalization(epsilon=epsilon, name=f"{name}_layernorm")(residual_layer)


def add_encoder_layer(
    input_layer,
    name,
    attention_activation=None,
    dropout=0.1,
    epsilon=1e-6,
    ff_activation="relu",
    ff_units=1000,
    num_heads=8,
):
    attention_layer = wrap_residual_with_dropout(
        input_layer,
        f"{name}_mha",
        MultiHeadAttention,
        dropout,
        epsilon,
        head_num=num_heads,
        activation=attention_activation,
        history_only=False,
        trainable=True,
    )
    return wrap_residual_with_dropout(
        attention_layer,
        f"{name}_ff",
        FeedForward,
        dropout,
        epsilon,
        units=ff_units,
        activation=ff_activation,
        trainable=True,
    )


def add_position_encoding(input_layer, name):
    return TrigPosEmbedding(name=name, mode=TrigPosEmbedding.MODE_ADD)(input_layer)
