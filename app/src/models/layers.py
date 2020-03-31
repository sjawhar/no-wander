import logging
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    deserialize,
    Dropout,
    LSTM,
    MaxPooling1D,
)
from .attention import add_encoder_layer, add_position_encoding
from .constants import (
    LAYER_ENCODER,
    LAYER_POSITION_ENCODING,
    PARAM_ACTIVATION,
    PARAM_NAME,
    PARAM_POOL_SIZE,
    PARAM_RETURN_SEQUENCES,
)

logger = logging.getLogger(__name__)


def add_ic_layer(layer, name, dropout=0.2, batchnorm=True):
    if batchnorm:
        logger.debug(f'Adding layer "{name}" - BatchNorm')
        layer = BatchNormalization(name=f"{name}_batchnorm")(layer)
    if dropout:
        logger.debug(f'Adding layer "{name}" - Dropout: {dropout}')
        layer = Dropout(dropout, name=f"{name}_dropout")(layer)
    return layer


def get_regularizers(**kwargs):
    regularizers = {}
    for arg in kwargs:
        if not arg.endswith("_regularizer"):
            continue
        config = kwargs[arg]
        if config is None or isinstance(config, regularizers.Regularizer):
            continue
        reg = (
            regularizers.L1L2(**config)
            if type(config) is dict
            else regularizers.get(config)
        )
        logger.debug(
            f"Adding regularizer {reg.__class__.__name__} to {name}: {reg.get_config()}"
        )
        regularizers[arg] = reg
    return regularizers


def add_layer_by_type(input_layer, layer_type, **kwargs):
    if type(layer_type) is type:
        return layer_type(**kwargs)(input_layer)

    return deserialize({"class_name": layer_type, "config": kwargs})(input_layer)


def add_conv1d_layer(input_layer, name, pool=None, **kwargs):
    kwargs.setdefault(PARAM_ACTIVATION, "relu")
    conv_layer = add_layer_by_type(input_layer, Conv1D, name=name, **kwargs)

    if pool is True:
        pool = {}
    elif type(pool) is int:
        pool = {PARAM_POOL_SIZE: pool}

    if type(pool) is not dict:
        return conv_layer

    return add_layer_by_type(conv_layer, MaxPooling1D, name=f"{name}_pool", **pool)


def get_layer_builder(layer_type):
    if layer_type == LAYER_ENCODER:
        return add_encoder_layer
    elif layer_type == LAYER_POSITION_ENCODING:
        return add_position_encoding
    elif type(layer_type) is Conv1D or layer_type == Conv1D.__name__:
        return add_conv1d_layer
    return lambda input_layer, name, **kwargs: add_layer_by_type(
        input_layer, layer_type, name=name, **kwargs
    )


def add_layer(
    input_layer, name, layer_type, is_last_of_type=False, ic_params=None, **layer_params
):
    if type(layer_type) is Dense or layer_type == Dense.__name__:
        layer_params.setdefault(PARAM_ACTIVATION, "relu")
    elif type(layer_type) is LSTM or layer_type == LSTM.__name__:
        layer_params.setdefault(PARAM_RETURN_SEQUENCES, not is_last_of_type)

    layer_type_name = layer_type.__name__ if type(layer_type) is type else layer_type
    logger.debug(f'Adding layer "{name}" - {layer_type_name}: {layer_params}')

    layer_builder = get_layer_builder(layer_type)
    layer_params.update(get_regularizers(**layer_params))
    next_layer = layer_builder(input_layer, name, **layer_params)

    if type(ic_params) is dict:
        next_layer = add_ic_layer(next_layer, name, **ic_params)

    return next_layer
