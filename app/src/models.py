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


def add_lstm_layer(model, name, params):
    ic_params = params.pop("ic_params", {})
    model.add(LSTM(name=name, **params))
    add_ic_layer(model, name, **ic_params)


def get_lstm_model(
    input_shape, lstm_layers, conv1d_params=None, dense_params=None,
):
    model = Sequential()

    if conv1d_params is not None:
        name = "conv1d"
        model.add(
            Conv1D(
                name=name, activation="relu", input_shape=input_shape, **conv1d_params
            )
        )
        add_ic_layer(model, name)

    num_lstm_layers = len(lstm_layers)
    for i in range(num_lstm_layers):
        name = f"lstm_{i + 1}"
        params = lstm_layers[i]
        if i == 0 and conv1d_params is None:
            params["input_shape"] = input_shape
        if i < num_lstm_layers - 1:
            params["return_sequences"] = True
        add_lstm_layer(model, name, params)

    name = "dense"
    if dense_params is None:
        dense_params = {}
    model.add(Dense(dense_params.get("units", 32), activation="relu", name=name))
    add_ic_layer(model, name, **dense_params.get("ic_params", {"dropout": 0.2}))

    model.add(Dense(1, activation="sigmoid", name="output"))

    return model
