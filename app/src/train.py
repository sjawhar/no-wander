import logging
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
    input_shape = (sequence_size, sample_size * len(features))

    logger.info(f"{len(samples)} samples in training set")
    logger.info(f"All seen features: {', '.join(features)}")

    if extract_features:
        samples, features = extract_eeg_features(X, features)
        logger.info(f"Extracted features: {', '.join(features)}")

    return samples, labels, features


def get_sequences(samples, labels, input_shape, shuffle_samples):
    if shuffle_samples:
        shuffle = np.random.permutation(samples.shape[0])
        samples = samples[shuffle]
        labels = labels[shufle]

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
    return np.nan_to_num(X, 0), Y


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

    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()

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
    model.fit(
        X, Y, epochs=epochs, batch_size=batch_size, **fit_args,
    )
