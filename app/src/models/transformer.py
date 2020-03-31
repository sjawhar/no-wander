# All glory to https://www.tensorflow.org/tutorials/text/transformer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization


def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)

    Returns:
    output, attention_weights
    """

    depth_k = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    logits /= tf.math.sqrt(depth_k)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    weights = tf.nn.softmax(logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(weights, v)  # (..., seq_len_q, depth_v)

    return output, weights


def split_heads(layer, batch_size, num_heads, depth):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, sequence_size, depth)
    """
    layer = tf.reshape(layer, (batch_size, -1, num_heads, depth))
    return tf.transpose(layer, perm=[0, 2, 1, 3])


def get_multihead_attention_layer(
    input_layer, name, num_heads=8, dim_model=512, activation=None
):
    assert dim_model % num_heads == 0
    depth = dim_model // num_heads
    batch_size = tf.shape(input_layer)[0]

    # (batch_size, sequence_size, dim_model)
    q = Dense(dim_model, name=f"{name}_query", activation=activation)(input_layer)
    k = Dense(dim_model, name=f"{name}_key", activation=activation)(input_layer)
    v = Dense(dim_model, name=f"{name}_value", activation=activation)(input_layer)

    # (batch_size, num_heads, sequence_size, depth)
    q = split_heads(q, batch_size, num_heads, depth)
    k = split_heads(k, batch_size, num_heads, depth)
    v = split_heads(v, batch_size, num_heads, depth)

    # (batch_size, num_heads, sequence_size, depth)
    attention, _ = scaled_dot_product_attention(q, k, v)
    # (batch_size, sequence_size, num_heads, depth)
    attention = tf.transpose(attention, perm=[0, 2, 1, 3])
    # (batch_size, sequence_size, dim_model)
    attention = tf.reshape(attention, (batch_size, -1, dim_model))

    # (batch_size, sequence_size, dim_model)
    return Dense(dim_model, name=f"{name}_output", activation=activation)(attention)


def get_feedforward_layer(
    input_layer, name, units_output, units_ff=1000, activation="relu"
):
    ff_layer = Dense(units_ff, activation=activation, name=f"{name}_ff")(input_layer)
    return Dense(units_output, name=f"{name}_output")(ff_layer)


def wrap_layer_with_res_and_dropout(
    input_layer, name, builder, dropout, epsilon, **kwargs
):
    output_layer = builder(input_layer, name, **kwargs)
    if dropout:
        output_layer = Dropout(rate=dropout, name=f"{name}_dropout")(output_layer)
    return LayerNormalization(epsilon=epsilon, name=f"{name}_layernorm")(
        input_layer + output_layer
    )


def add_encoder_layer(
    input_layer,
    name,
    activation_attention=None,
    activation_ff="relu",
    dim_model=512,
    dropout=0.0,
    epsilon=1e-6,
    num_heads=8,
    units_ff=1000,
):
    attention_layer = wrap_layer_with_res_and_dropout(
        input_layer,
        f"{name}_mha",
        get_multihead_attention_layer,
        dropout,
        epsilon,
        activation=activation_attention,
        dim_model=dim_model,
        num_heads=num_heads,
    )
    return wrap_layer_with_res_and_dropout(
        attention_layer,
        f"{name}_ff",
        get_feedforward_layer,
        dropout,
        epsilon,
        activation=activation_ff,
        units_ff=units_ff,
        units_output=dim_model,
    )
