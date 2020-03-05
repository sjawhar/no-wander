from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Input,
    LSTM,
    BatchNormalization,
    Conv1D,
)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

PARAM_RETURN_SEQUENCES = "return_sequences"


def add_ic_layer(model, name, dropout=0.2, batchnorm=True):
    if batchnorm:
        model.add(BatchNormalization(name=f"{name}_bn"))
    if dropout:
        model.add(Dropout(dropout, name=f"{name}_dropout"))


def add_conv_layer(model, name, activation="relu", ic_params={}, **kwargs):
    model.add(Conv1D(name=name, activation=activation, **kwargs))
    add_ic_layer(model, name, **ic_params)


def add_lstm_layer(model, name, ic_params={}, **kwargs):
    model.add(LSTM(name=name, **kwargs))
    add_ic_layer(model, name, **ic_params)


def add_dense_layer(model, name, units=32, activation="relu", ic_params={}, **kwargs):
    model.add(Dense(units, activation=activation, name=name, **kwargs))
    add_ic_layer(model, name, **ic_params)


def compile_model(
    model, learning_rate, beta_one, beta_two, decay,
):
    opt = Adam(learning_rate, beta_one, beta_two, decay)
    model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])


def get_lstm_model(
    input_shape, lstm_layers, conv1d_params=None, dense_params={}, plot_model_file=None,
):
    model = Sequential()

    if conv1d_params is not None:
        add_conv_layer(model, "conv1d", input_shape=input_shape, **conv1d_params)

    if type(lstm_layers) is not list:
        lstm_layers = [lstm_layers]

    num_lstm_layers = len(lstm_layers)
    for i in range(num_lstm_layers):
        name = f"lstm_{i + 1}"
        lstm_params = lstm_layers[i]
        if i == 0 and conv1d_params is None:
            lstm_params["input_shape"] = input_shape
        if PARAM_RETURN_SEQUENCES not in lstm_params:
            lstm_params[PARAM_RETURN_SEQUENCES] = i < num_lstm_layers - 1
        add_lstm_layer(model, name, **lstm_params)

    add_dense_layer(model, "dense", **dense_params)
    model.add(Dense(1, activation="sigmoid", name="output"))

    if plot_model_file is not None:
        plot_model(
            model, to_file=plot_model_file, show_shapes=True, show_layer_names=True
        )

    return model
