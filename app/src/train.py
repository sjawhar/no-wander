import logging
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model
from pathlib import Path
from sklearn.model_selection import train_test_split
from .datasets import read_dataset
from .features import preprocess_data
from .models import get_lstm_model
from .constants import PACKAGE_NAME, PREPROCESS_NONE

LEARNING_RATE = 0.1
BETA_ONE = 0.9
BETA_TWO = 0.999
DECAY = 0.01
RANDOM_SEED = 42

logger = logging.getLogger(PACKAGE_NAME + "." + __name__)


def get_sequences(samples, labels, input_shape, shuffle_samples):
    logger.info(f"Forming sequences of length {input_shape[0]}...")
    if shuffle_samples:
        logger.debug("Shuffling samples...")
        np.random.seed(RANDOM_SEED)
        shuffle = np.random.permutation(samples.shape[0])
        samples = samples[shuffle]
        labels = labels[shuffle]

    X = []
    for label in [0, 1]:
        X_label = samples[(labels == label).flatten()]
        rem = X_label.shape[0] % input_shape[0]
        if rem > 0:
            X_label = X_label[:-rem]
        X.append(X_label.reshape((-1, *input_shape)))

    miss_size = X[0].shape[0]
    X = np.concatenate(X)
    Y = np.ones((X.shape[0], 1))
    Y[:miss_size] = 0
    logger.info(f"Formed {Y.size} sequences! {miss_size} miss / {int(Y.sum())} hit")
    return np.nan_to_num(X, 0), Y


def train_model(
    model,
    X,
    Y,
    test_split=0,
    learning_rate=LEARNING_RATE,
    beta_one=BETA_ONE,
    beta_two=BETA_TWO,
    decay=DECAY,
    **kwargs,
):
    validation_data = None
    if test_split:
        logger.info(f"Splitting {int(test_split * 100)}% of data for validation")
        X, X_test, Y, Y_test = train_test_split(
            X, Y, test_size=test_split, random_state=RANDOM_SEED
        )
        validation_data = (X_test, Y_test)

    opt = Adam(learning_rate, beta_one, beta_two, decay)
    model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model.fit(X, Y, validation_data=validation_data, **kwargs)


def plot_training_history(history, model_dir):
    import matplotlib.pyplot as plt

    for metric in ["accuracy", "loss"]:
        plt.plot(history.history[metric])
        plt.title(f"Model {metric}")
        plt.ylabel(metric)
        plt.xlabel("epoch")

        legend = ["Train"]
        val_metric = history.history.get(f"val_{metric}", None)
        if val_metric is not None:
            plt.plot(val_metric)
            legend.append("Validation")
        plt.legend(legend, loc="upper left")

        figpath = model_dir / f"{metric}.png"
        logger.info(f"Saving {metric} chart to {figpath}...")
        plt.savefig(figpath, bbox_inches="tight")
        plt.close()


def build_and_train_model(
    data_file,
    model_dir,
    # Model params
    sample_size,
    sequence_size,
    lstm_layers,
    conv1d_params=None,
    dense_params={},
    # Data prep params
    preprocess=PREPROCESS_NONE,
    shuffle_samples=False,
    **train_args,
):
    samples, labels, features_raw = read_dataset(data_file, sample_size)
    logger.info(f"{len(samples)} samples in training set")
    logger.info(f"Raw features: {', '.join(features_raw)}")

    samples, sample_dims, features = preprocess_data(samples, features_raw, preprocess)
    input_shape = (sequence_size, sample_dims)
    logger.info(f"Preprocessed features: {', '.join(features)}")

    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        model_dir.mkdir()
    logger.debug(f"Saving model files to {model_dir}")

    model = get_lstm_model(
        input_shape,
        lstm_layers,
        conv1d_params=conv1d_params,
        dense_params=dense_params,
    )
    model.summary()
    plot_model(
        model, to_file=model_dir / "model.png", show_shapes=True, show_layer_names=True
    )

    if train_args.get("epochs"):
        X, Y = get_sequences(samples, labels, input_shape, shuffle_samples)

        history = train_model(model, X, Y, **train_args)
        plot_training_history(history, model_dir)

    model_path = model_dir / "model.h5"
    logger.info(f"Saving model to {model_path}...")
    model.save(model_path)
    logger.info("Done!")
