import logging
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model
from pathlib import Path
from .datasets import read_dataset
from .models import get_lstm_model
from .constants import PACKAGE_NAME

LEARNING_RATE = 0.1
BETA_ONE = 0.9
BETA_TWO = 0.999
DECAY = 0.01

logger = logging.getLogger(PACKAGE_NAME + "." + __name__)


def get_samples(data_file, sample_size, extract_features):
    samples, labels, features = read_dataset(data_file, sample_size)

    logger.info(f"{len(samples)} samples in training set")
    logger.info(f"All seen features: {', '.join(features)}")

    if extract_features:
        from .extract import extract_eeg_features

        samples, features = extract_eeg_features(samples, features)
        logger.info(f"Extracted features: {', '.join(features)}")

    return samples, labels, features


def get_sequences(samples, labels, input_shape, shuffle_samples):
    logger.info(f"Forming sequences of length {input_shape[0]}...")
    if shuffle_samples:
        logger.debug("Shuffling samples...")
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


def plot_history(history, model_dir):
    import matplotlib.pyplot as plt

    for metric in ["accuracy", "loss"]:
        plt.plot(history.history[metric])
        plt.plot(history.history[f"val_{metric}"])
        plt.title(f"Model {metric}")
        plt.ylabel(metric)
        plt.xlabel("epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
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
    dense_params=None,
    extract_features=False,
    # Learning rate params
    learning_rate=LEARNING_RATE,
    beta_one=BETA_ONE,
    beta_two=BETA_TWO,
    decay=DECAY,
    # Fit params
    shuffle_samples=False,
    epochs=100,
    batch_size=32,
    **fit_args,
):
    samples, labels, features = get_samples(data_file, sample_size, extract_features)
    input_shape = (
        sequence_size,
        len(features) * (1 if extract_features else sample_size),
    )

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

    opt = Adam(learning_rate, beta_one, beta_two, decay)
    model.compile(opt, loss="binary_crossentropy", metrics=["accuracy"])

    X, Y = get_sequences(samples, labels, input_shape, shuffle_samples)
    if epochs:
        history = model.fit(
            X,
            Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.25,
            **fit_args,
        )
        plot_history(history, model_dir)

    model_path = model_dir / "model.h5"
    logger.info(f"Saving model to {model_path}...")
    model.save(model_path)
    logger.info("Done!")
