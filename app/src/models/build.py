import json
import logging
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    deserialize,
    Dropout,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from .layers import add_layer
from .constants import (
    LAYER_POSITION_ENCODING,
    PARAM_ACTIVATION,
    PARAM_NAME,
    PARAM_UNITS,
)

logger = logging.getLogger(__name__)


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


def count_layers(layers):
    last_layers = {}
    layer_count = []
    num_layers = len(layers)
    for i in range(num_layers):
        layer_type = layers[i].get("type")
        last_layer_index, last_layer_num = last_layers.get(layer_type, (None, 0))

        layer_num = last_layer_num + 1
        layer_count.append([layer_num, True])
        if last_layer_index is not None:
            layer_count[last_layer_index][1] = False

        last_layers[layer_type] = (i, layer_num)

    return layer_count


def get_model_from_layers(
    layers,
    input_shape,
    dropout=0,
    encode_position=False,
    output={},
    plot_model_file=None,
):
    if type(layers) is not list:
        layers = [layers]
    elif len(layers) == 0:
        raise ValueError("Must specify at least one layer")

    input_layer = Input(shape=input_shape, name="input")
    X = input_layer
    if encode_position:
        X = add_layer(X, LAYER_POSITION_ENCODING, "input_encode_position")
    if dropout > 0:
        noise_shape = [None, *input_shape]
        if layers[0]["type"] == LSTM.__name__:
            noise_shape[1] = 1
        X = add_layer(
            X,
            Dropout,
            "input_dropout",
            ic_params=None,
            noise_shape=noise_shape,
            rate=dropout,
        )

    layer_count = count_layers(layers)
    for i in range(len(layers)):
        layer_params = layers[i]
        layer_type = layer_params.pop("type")
        layer_type_num, is_last_of_type = layer_count[i]

        name = layer_params.pop(PARAM_NAME, f"{layer_type}_{layer_type_num}".lower())
        X = add_layer(
            X, layer_type, name, is_last_of_type=is_last_of_type, **layer_params,
        )

    if not output:
        output = {}
    output.setdefault(PARAM_UNITS, 1)
    output.setdefault(PARAM_ACTIVATION, "sigmoid")
    output_layer = add_layer(X, Dense, "output", ic_params=None, **output)

    model = Model(inputs=input_layer, outputs=output_layer)
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
