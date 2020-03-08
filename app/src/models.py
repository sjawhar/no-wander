import logging
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    LSTM,
    BatchNormalization,
    Conv1D,
)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

logger = logging.getLogger(__name__)

PARAM_ACTIVATION = "activation"
PARAM_INPUT_SHAPE = "input_shape"
PARAM_RETURN_SEQUENCES = "return_sequences"
PARAM_UNITS = "units"


def add_ic_layer(model, name, dropout=0.2, batchnorm=True):
    if batchnorm:
        logger.debug(f'Adding layer "{name}" - BatchNorm')
        model.add(BatchNormalization(name=f"{name}_bn"))
    if dropout:
        logger.debug(f'Adding layer "{name}" - Dropout: {dropout}')
        model.add(Dropout(dropout, name=f"{name}_dropout"))


def add_conv_layer(model, name, ic_params={}, **kwargs):
    logger.debug(f'Adding layer "{name}" - Conv1D: {kwargs}')
    model.add(Conv1D(name=name, **kwargs))
    add_ic_layer(model, name, **ic_params)


def add_lstm_layer(model, name, ic_params={}, **kwargs):
    logger.debug(f'Adding layer "{name}" - LSTM: {kwargs}')
    model.add(LSTM(name=name, **kwargs))
    add_ic_layer(model, name, **ic_params)


def add_dense_layer(model, name, ic_params={}, **kwargs):
    logger.debug(f'Adding layer "{name}" - Dense: {kwargs}')
    model.add(Dense(name=name, **kwargs))
    add_ic_layer(model, name, **ic_params)


def compile_model(
    model, learning_rate, beta_one, beta_two, decay,
):
    opt = Adam(learning_rate, beta_one, beta_two, decay)
    model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])


def get_lstm_model(
    input_shape, lstm_layers, conv1d_layers=[], dense_params={}, plot_model_file=None,
):
    model = Sequential()
    is_input_layer = True

    if type(conv1d_layers) is not list:
        conv1d_layers = [conv1d_layers]
    for i in range(len(conv1d_layers)):
        name = f"conv1d_{i + 1}"
        conv1d_params = conv1d_layers[i]
        if is_input_layer:
            conv1d_params[PARAM_INPUT_SHAPE] = input_shape
            is_input_layer = False
        conv1d_params[PARAM_ACTIVATION] = conv1d_params.get(PARAM_ACTIVATION, "relu")
        add_conv_layer(model, name, **conv1d_params)

    if type(lstm_layers) is not list:
        lstm_layers = [lstm_layers]
    num_lstm_layers = len(lstm_layers)
    for i in range(num_lstm_layers):
        name = f"lstm_{i + 1}"
        lstm_params = lstm_layers[i]
        if is_input_layer:
            lstm_params[PARAM_INPUT_SHAPE] = input_shape
            is_input_layer = False
        if PARAM_RETURN_SEQUENCES not in lstm_params:
            lstm_params[PARAM_RETURN_SEQUENCES] = i < num_lstm_layers - 1
        add_lstm_layer(model, name, **lstm_params)

    dense_params.update(
        {
            PARAM_UNITS: dense_params.get(PARAM_UNITS, 32),
            PARAM_ACTIVATION: dense_params.get(PARAM_ACTIVATION, "relu"),
        }
    )
    add_dense_layer(model, "dense", **dense_params)
    add_dense_layer(model, "output", **{PARAM_UNITS: 1, PARAM_ACTIVATION: "sigmoid"})

    if plot_model_file is not None:
        plot_model(
            model, to_file=plot_model_file, show_shapes=True, show_layer_names=True
        )

    return model
