import json
import logging
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
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
PARAM_RETURN_SEQUENCES = "return_sequences"
PARAM_UNITS = "units"
PARAM_VERBOSE = "verbose"


class GradientMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        if not hasattr(self, "is_chart_created"):
            self.is_chart_created = {}

        for metric, value in logs.items():
            if metric not in self.is_chart_created:
                print(json.dumps({"chart": metric, "axis": "epoch"}))
                self.is_chart_created[metric] = True
            print(json.dumps({"chart": metric, "x": epoch, "y": float(value)}))


def add_ic_layer(model, name, dropout=0.2, batchnorm=True):
    if batchnorm:
        logger.debug(f'Adding layer "{name}" - BatchNorm')
        model.add(BatchNormalization(name=f"{name}_bn"))
    if dropout:
        logger.debug(f'Adding layer "{name}" - Dropout: {dropout}')
        model.add(Dropout(dropout, name=f"{name}_dropout"))


def add_layer(model, layer, name, ic_params={}, **kwargs):
    logger.debug(f'Adding layer "{name}" - {layer.__name__}: {kwargs}')
    model.add(layer(name=name, **kwargs))
    if type(ic_params) is not dict:
        return
    add_ic_layer(model, name, **ic_params)


def add_conv1d_layer(model, name, ic_params={}, pool=None, **kwargs):
    add_layer(model, Conv1D, name, ic_params=None, **kwargs)
    if pool is not None:
        add_layer(model, MaxPooling1D, f"{name}_pool", ic_params=None, **pool)
    add_ic_layer(model, name, **ic_params)


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
        kwargs[PARAM_VERBOSE] = 0
    return model.fit(X, Y, callbacks=callbacks, **kwargs)


def get_lstm_model(
    input_shape,
    lstm_layers,
    conv1d_layers=[],
    dense_params={},
    dropout=0,
    plot_model_file=None,
):
    model = Sequential()
    is_input_layer = True

    if type(conv1d_layers) is not list:
        conv1d_layers = [conv1d_layers]
    if type(lstm_layers) is not list:
        lstm_layers = [lstm_layers]

    if dropout > 0:
        noise_shape = [None, *input_shape]
        if len(conv1d_layers):
            noise_shape[1] = 1
        add_layer(
            model,
            Dropout,
            "input_dropout",
            ic_params=None,
            input_shape=input_shape,
            noise_shape=noise_shape,
            rate=dropout,
        )
        is_input_layer = False

    for i in range(len(conv1d_layers)):
        name = f"conv1d_{i + 1}"
        conv1d_params = conv1d_layers[i]
        if is_input_layer:
            conv1d_params[PARAM_INPUT_SHAPE] = input_shape
            is_input_layer = False
        conv1d_params[PARAM_ACTIVATION] = conv1d_params.get(PARAM_ACTIVATION, "relu")
        add_conv1d_layer(model, name, **conv1d_params)

    num_lstm_layers = len(lstm_layers)
    for i in range(num_lstm_layers):
        name = f"lstm_{i + 1}"
        lstm_params = lstm_layers[i]
        if is_input_layer:
            lstm_params[PARAM_INPUT_SHAPE] = input_shape
            is_input_layer = False
        if PARAM_RETURN_SEQUENCES not in lstm_params:
            lstm_params[PARAM_RETURN_SEQUENCES] = i < num_lstm_layers - 1
        add_layer(model, LSTM, name, **lstm_params)

    dense_params.update(
        {
            PARAM_UNITS: dense_params.get(PARAM_UNITS, 32),
            PARAM_ACTIVATION: dense_params.get(PARAM_ACTIVATION, "relu"),
        }
    )
    add_layer(model, Dense, "dense", **dense_params)
    add_layer(
        model,
        Dense,
        "output",
        **{PARAM_UNITS: 1, PARAM_ACTIVATION: "sigmoid", "ic_params": None},
    )

    if plot_model_file is not None:
        plot_model(
            model, to_file=plot_model_file, show_shapes=True, show_layer_names=True
        )

    return model
