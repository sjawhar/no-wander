from keras.models import Sequential
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Input,
    LSTM,
    BatchNormalization,
    Conv1D,
)


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


def get_lstm_model(
    input_shape, lstm_layers, conv1d_params=None, dense_params={},
):
    model = Sequential()

    if conv1d_params is not None:
        add_conv_layer(model, "conv1d", input_shape=input_shape, **conv1d_params)

    num_lstm_layers = len(lstm_layers)
    for i in range(num_lstm_layers):
        name = f"lstm_{i + 1}"
        lstm_params = lstm_layers[i]
        if i == 0 and conv1d_params is None:
            lstm_params["input_shape"] = input_shape
        if i < num_lstm_layers - 1:
            lstm_params["return_sequences"] = True
        add_lstm_layer(model, name, **lstm_params)

    add_dense_layer(model, "dense", **dense_params)
    model.add(Dense(1, activation="sigmoid", name="output"))

    return model
