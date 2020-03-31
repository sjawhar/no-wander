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


def wrap_layer_with_ic(layer, name, dropout=0.2, batchnorm=True):
    if batchnorm:
        logger.debug(f'Adding layer "{name}" - BatchNorm')
        layer = BatchNormalization(name=f"{name}_batchnorm")(layer)
    if dropout:
        logger.debug(f'Adding layer "{name}" - Dropout: {dropout}')
        layer = Dropout(dropout, name=f"{name}_dropout")(layer)
    return layer


def add_wrapped_layer(input_layer, layer_type, name, ic_params={}, **kwargs):
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
        kwargs[arg] = reg

    layer = deserialize(
        {"class_name": layer_type, "config": {PARAM_NAME: name, **kwargs}}
    )(input_layer)
    if ic_params:
        layer = wrap_layer_with_ic(layer, name, **ic_params)
    return layer


def add_conv1d_layer(input_layer, name, ic_params={}, pool=None, **kwargs):
    kwargs.setdefault(PARAM_ACTIVATION, "relu")
    conv_layer = add_wrapped_layer(input_layer, Conv1D, name, ic_params=None, **kwargs)

    if pool:
        if pool is True:
            pool = {}
        elif type(pool) is int:
            pool = {PARAM_POOL_SIZE: pool}
        conv_layer = add_wrapped_layer(
            conv_layer, MaxPooling1D, f"{name}_pool", ic_params=None, **pool
        )

    if ic_params:
        conv_layer = wrap_layer_with_ic(conv_layer, name, **ic_params)

    return conv_layer


def add_layer(input_layer, layer_type, name, is_last_of_type=False, **layer_params):
    if type(layer_type) is type:
        layer_type = layer_type.__name__

    layer_builder = None
    if layer_type == LAYER_ENCODER:
        layer_builder = add_encoder_layer
    elif layer_type == LAYER_POSITION_ENCODING:
        layer_builder = add_position_encoding
    elif layer_type == Conv1D.__name__:
        layer_builder = add_conv1d_layer
    elif layer_type == Dense.__name__:
        layer_params.setdefault(PARAM_ACTIVATION, "relu")
    elif layer_type == LSTM.__name__:
        layer_params.setdefault(PARAM_RETURN_SEQUENCES, not is_last_of_type)

    logger.debug(f'Adding layer "{name}" - {layer_type}: {layer_params}')
    if layer_builder is not None:
        return layer_builder(input_layer, name, **layer_params)
    return add_wrapped_layer(input_layer, layer_type, name, **layer_params)
