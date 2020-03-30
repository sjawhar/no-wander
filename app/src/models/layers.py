import json
import logging
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    deserialize,
    Dropout,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

logger = logging.getLogger(__name__)

PARAM_ACTIVATION = "activation"
PARAM_INPUT_SHAPE = "input_shape"
PARAM_NAME = "name"
PARAM_POOL_SIZE = "pool_size"
PARAM_RETURN_SEQUENCES = "return_sequences"
PARAM_UNITS = "units"
PARAM_VERBOSE = "verbose"


class GradientMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        if not hasattr(self, "is_chart_created"):
            self.is_chart_created = {}

        print("")
        for metric, value in logs.items():
            if metric not in self.is_chart_created:
                print(json.dumps({"chart": metric, "axis": "epoch"}))
                self.is_chart_created[metric] = True
            print(json.dumps({"chart": metric, "x": epoch, "y": float(value)}))
        print("")


def get_ic_layer(name, dropout=0.2, batchnorm=True):
    layers = []
    if batchnorm:
        logger.debug(f'Adding layer "{name}" - BatchNorm')
        layers.append(BatchNormalization(name=f"{name}_bn"))
    if dropout:
        logger.debug(f'Adding layer "{name}" - Dropout: {dropout}')
        layers.append(Dropout(dropout, name=f"{name}_dropout"))
    return layers


def get_wrapped_layer(layer_type, name, ic_params={}, **kwargs):
    if type(layer_type) is type:
        layer_type = layer_type.__name__
    logger.debug(f'Adding layer "{name}" - {layer_type}: {kwargs}')

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

    layers = [
        deserialize({"class_name": layer_type, "config": {PARAM_NAME: name, **kwargs}})
    ]
    if ic_params:
        layers += get_ic_layer(name, **ic_params)
    return layers


def get_conv1d_layer(name, ic_params={}, pool=None, **kwargs):
    kwargs.setdefault(PARAM_ACTIVATION, "relu")
    layers = get_wrapped_layer(Conv1D, name, ic_params=ic_params, **kwargs)
    if not pool:
        return layers

    if pool is True:
        pool = {}
    elif type(pool) is int:
        pool = {PARAM_POOL_SIZE: pool}
    layers.insert(
        -1,
        get_wrapped_layer(MaxPooling1D, name=f"{name}_pool", ic_params=None, **pool)[0],
    )
    return layers


def get_layer(layer_type, name, is_last_of_type=False, **layer_params):
    layer = None
    if layer_type == Conv1D.__name__:
        return get_conv1d_layer(name, **layer_params)
    elif layer_type == LSTM.__name__:
        layer_params.setdefault(PARAM_RETURN_SEQUENCES, not is_last_of_type)
    elif layer_type == Dense.__name__:
        layer_params.setdefault(PARAM_ACTIVATION, "relu")
    return get_wrapped_layer(layer_type, name, **layer_params)


def count_layers(layers, default_type=None):
    last_layers = {}
    layer_count = []
    num_layers = len(layers)
    for i in range(num_layers):
        layer_type = layers[i].get("type", default_type)
        if layer_type is None:
            return ValueError(f"layer {i} has no type")
        last_layer_index, last_layer_num = last_layers.get(layer_type, (None, 0))

        layer_num = last_layer_num + 1
        layer_count.append([layer_num, True])
        if last_layer_index is not None:
            layer_count[last_layer_index][1] = False

        last_layers[layer_type] = (i, layer_num)

    return layer_count


def get_model_layers(layers, default_type=None, input_shape=None):
    model_layers = []
    layer_count = count_layers(layers, default_type=default_type)
    for i in range(len(layers)):
        layer_params = layers[i]
        layer_type = layer_params.pop("type", default_type)
        layer_type_num, is_last_of_type = layer_count[i]

        name = layer_params.pop(PARAM_NAME, f"{layer_type}_{layer_type_num}".lower())
        if input_shape is not None:
            layer_params[PARAM_INPUT_SHAPE] = input_shape
            input_shape = None
        model_layers += get_layer(
            layer_type, name, is_last_of_type=is_last_of_type, **layer_params,
        )

    return model_layers


def get_model_from_layers(
    layers, input_shape, dropout=0, output={}, plot_model_file=None,
):
    if type(layers) is not list:
        layers = [layers]
    elif len(layers) == 0:
        raise ValueError("Must specify at least one layer")

    model_layers = []
    if dropout > 0:
        noise_shape = [None, *input_shape]
        if layers[0]["type"] == LSTM.__name__:
            noise_shape[1] = 1
        model_layers = get_wrapped_layer(
            Dropout,
            "input_dropout",
            ic_params=None,
            input_shape=input_shape,
            noise_shape=noise_shape,
            rate=dropout,
        )
        input_shape = None

    model_layers += get_model_layers(layers, input_shape=input_shape)

    if output is not None:
        output.setdefault(PARAM_UNITS, 1)
        output.setdefault(PARAM_ACTIVATION, "sigmoid")
        model_layers += get_wrapped_layer(Dense, "output", ic_params=None, **output)

    model = Sequential(model_layers)
    if plot_model_file is not None:
        plot_model(
            model, to_file=plot_model_file, show_shapes=True, show_layer_names=True
        )

    return model


def compile_model(model, learning_rate, beta_one, beta_two, decay):
    opt = Adam(learning_rate, beta_one, beta_two, decay)
    model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])


def fit_model(
    model,
    X,
    Y,
    checkpoint_path=None,
    tensorboard_path=None,
    gradient_metrics=False,
    **kwargs,
):
    callbacks = []
    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path)
        logger.debug(f"Model with best val_loss will be saved to {checkpoint_path}")
        callbacks.append(
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")
        )
    if tensorboard_path is not None:
        tensorboard_path = str(tensorboard_path)
        logger.debug(f"TensorBoard logs will be saved to {tensorboard_path}")
        callbacks.append(
            TensorBoard(tensorboard_path, histogram_freq=100, write_images=True)
        )
    if gradient_metrics:
        callbacks.append(GradientMetricsCallback())
    return model.fit(X, Y, callbacks=callbacks, **kwargs)
