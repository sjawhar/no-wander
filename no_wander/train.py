from pathlib import Path
import logging
import pickle

from sklearn.model_selection import train_test_split
import numpy as np

from .datasets import read_dataset, DATASET_TRAIN, DATASET_VAL
from .features import preprocess_data_test, preprocess_data_train
from .models import compile_model, fit_model, get_model_from_layers
from .constants import PREPROCESS_NONE, RANDOM_SEED

LEARNING_RATE = 0.1
BETA_ONE = 0.9
BETA_TWO = 0.999
DECAY = 0.01
WINDOW_POST_RECOVERY = (0, 3)
WINDOW_PRE_RECOVERY = (-7, -1)

logger = logging.getLogger(__name__)


def get_sequences(segments, labels, input_shape, shuffle_segments):
    sequence_size = input_shape[0]
    logger.info(f"Forming sequences of length {sequence_size}...")
    if shuffle_segments:
        logger.debug("Shuffling segments...")
        np.random.seed(RANDOM_SEED)
        shuffle = np.random.permutation(segments.shape[0])
        segments = segments[shuffle]
        labels = labels[shuffle]

    X = []
    for label, label_name in [(0, "miss"), (1, "hit")]:
        X_label = segments[(labels == label).flatten()]
        num_segments = X_label.shape[0]
        rem = num_segments % sequence_size
        if rem > 0:
            if shuffle_segments:
                pad = np.random.choice(
                    num_segments, size=(sequence_size - rem), replace=False
                )
                X_label = np.concatenate([X_label, X_label[pad]], axis=0)
                logger.debug(
                    f"Padded {label_name} with {len(pad)} segments for even sequences"
                )
            else:
                X_label = X_label[:-rem]
                logger.debug(f"Dropped {rem} {label_name} segments for even sequences")
        X.append(X_label.reshape((-1, *input_shape)))

    miss_size = X[0].shape[0]
    X = np.concatenate(X)
    Y = np.ones((X.shape[0], 1))
    Y[:miss_size] = 0
    logger.info(f"Formed {Y.size} sequences! {miss_size} miss / {int(Y.sum())} hit")
    logger.debug(f"Sequences shape: {X.shape}")
    return np.nan_to_num(X, 0), Y


def plot_training_history(history, model_dir):
    import matplotlib.pyplot as plt

    for metric in ["accuracy", "loss"]:
        metric_value = history.history.get(metric, None)
        if not metric_value:
            continue
        plt.plot(metric_value)
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


def train_model(
    model,
    model_dir,
    X_train,
    Y_train,
    X_val,
    Y_val,
    input_shape,
    shuffle_segments=False,
    # Optimizer params
    learning_rate=LEARNING_RATE,
    beta_one=BETA_ONE,
    beta_two=BETA_TWO,
    decay=DECAY,
    # Logging params
    checkpoint=True,
    gradient_metrics=False,
    tensorboard=False,
    **kwargs,
):
    compile_model(model, learning_rate, beta_one, beta_two, decay)

    X, Y = get_sequences(X_train, Y_train, input_shape, shuffle_segments)
    validation_data = get_sequences(X_val, Y_val, input_shape, False)
    logger.info(
        f"Train on {len(X)} samples, validate on {len(validation_data[0])} samples"
    )

    try:
        history = fit_model(
            model,
            X,
            Y,
            checkpoint_path=model_dir / "model_best" if checkpoint else None,
            gradient_metrics=gradient_metrics,
            tensorboard_path=model_dir / "tensorboard" if tensorboard else None,
            validation_data=validation_data,
            **kwargs,
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted!")
        history = model.history

    plot_training_history(history, model_dir)
    return history


def build_and_train_model(
    data_file,
    model_dir,
    # Model params
    segment_size,
    sequence_size,
    layers,
    dropout=0,
    encode_position=False,
    output={},
    plot_model=True,
    # Data prep params
    post_window=WINDOW_POST_RECOVERY,
    pre_window=WINDOW_PRE_RECOVERY,
    preprocess=PREPROCESS_NONE,
    shuffle_segments=False,
    **train_kwargs,
):
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Model files will be saved to {model_dir}")

    datasets, features_raw = read_dataset(
        data_file,
        segment_size,
        1 if shuffle_segments else sequence_size,
        pre_window,
        post_window,
    )
    samples_train, labels_train = datasets[DATASET_TRAIN]
    logger.info(f"{len(samples_train)} samples in training set")
    logger.info(f"Raw features: {', '.join(features_raw)}")

    samples_train, preprocessor, features, is_flattened = preprocess_data_train(
        samples_train, preprocess, features_raw
    )
    input_shape = (sequence_size, len(features) * (1 if is_flattened else segment_size))
    logger.info(f"Preprocessed features: {', '.join(features)}")
    logger.info(f"Input shape: {input_shape}")

    if preprocessor is not None:
        preprocess_path = str(model_dir / f"preprocess.pickle")
        logger.info(f"Saving preprocessing details to {preprocess_path}...")
        with open(preprocess_path, "wb") as f:
            pickle.dump(preprocessor, f)

    model = get_model_from_layers(
        layers,
        input_shape,
        dropout=dropout,
        encode_position=encode_position,
        output=output,
        plot_model_file=model_dir / "model.png" if plot_model is True else None,
    )
    model.summary()

    if train_kwargs.get("epochs", 0):
        samples_val, labels_val = datasets[DATASET_VAL]
        samples_val = preprocess_data_test(samples_val, preprocessor)
        train_model(
            model,
            model_dir,
            samples_train,
            labels_train,
            samples_val,
            labels_val,
            input_shape,
            shuffle_segments=shuffle_segments,
            **train_kwargs,
        )

    model_path = str(model_dir / "model_final")
    logger.info(f"Saving model to {model_path}...")
    model.save(model_path)
    logger.info("Done!")
