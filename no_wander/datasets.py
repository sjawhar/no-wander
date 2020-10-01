import h5py
import logging
import numpy as np
from .constants import (
    COL_MARKER_DEFAULT,
    DATASET_TEST,
    DATASET_TRAIN,
    DATASET_VAL,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


def parse_dataset(filepath, train_set=DATASET_TRAIN, test_set=DATASET_TEST):
    datasets = {}
    features = {}

    with h5py.File(filepath, "r") as hf:
        for set_type in [train_set, test_set]:
            if set_type is None:
                continue

            set_group = hf[set_type]
            data = [None] * len(set_group)
            for i, dset in enumerate(set_group.values()):
                columns = dset.attrs["columns"]
                for col in columns:
                    if col not in features:
                        features[col] = len(features)
                column_numbers = [features[col] for col in columns]

                data[i] = dset[:], dset.attrs["recovery"], column_numbers
            datasets[set_type] = data

    features = [feat for (feat, _) in sorted(features.items(), key=lambda x: x[1])]
    return datasets, features


def get_window_segments(data, segment_size, consecutive_segments, window):
    start, stop = window
    is_flip = stop <= 0
    if is_flip:
        start, stop = -stop, -start
        data = data[::-1]

    data = data[start:stop]
    drop = len(data) % (segment_size * consecutive_segments)
    if drop > 0:
        data = data[:-drop]

    segments = data.reshape(-1, segment_size, data.shape[1])
    if is_flip:
        segments = segments[::-1, ::-1]
    return list(segments), drop


def get_segments(data, segment_size, consecutive_segments, pre_window, post_window):
    parsed = [None] * len(data)
    num_segments = 0

    for i, (dset, recovery_ix, features) in enumerate(data):
        parsed[i] = [[], [], features]
        for j, window_data, window, window_name in [
            (0, dset[:recovery_ix], pre_window, "pre"),
            (1, dset[recovery_ix:], post_window, "post"),
        ]:
            window_segments, dropped = get_window_segments(
                window_data, segment_size, consecutive_segments, window
            )
            parsed[i][j] = window_segments
            if dropped > 0:
                logger.debug(f"Dropped {dropped} readings from {window_name} window")
            num_segments += len(window_segments)

    return parsed, num_segments


def segments_to_tensors(segments, data_shape):
    X = np.full(data_shape, np.nan)
    Y = np.zeros((X.shape[0], 1))

    i = 0
    for pre, post, features in segments:
        for window, label in [(pre, 0), (post, 1)]:
            window_size = len(window)
            if window_size == 0:
                continue
            i_next = i + window_size
            X[i:i_next, :, features] = window
            Y[i:i_next] = label
            i = i_next

    return X, Y


def read_dataset(
    filepath,
    segment_size,
    consecutive_segments,
    pre_window,
    post_window,
    sample_rate=SAMPLE_RATE,
):
    logger.info(f"Reading datasets from {filepath}...")
    pre_window = tuple(int(ix * sample_rate) for ix in pre_window)
    post_window = tuple(int(ix * sample_rate) for ix in post_window)

    data, features = parse_dataset(filepath, test_set=DATASET_VAL)
    datasets = {}
    for set_type in [DATASET_TRAIN, DATASET_VAL]:
        segments, num_segments = get_segments(
            data[set_type], segment_size, consecutive_segments, pre_window, post_window
        )
        X, Y = segments_to_tensors(
            segments, (num_segments, segment_size, len(features))
        )
        logger.debug(f"{set_type} data shape {X.shape}")
        datasets[set_type] = (X, Y)
    return datasets, features


def save_epochs(filepath, epochs):
    logger.info(f"Saving epochs to {filepath}...")
    if type(epochs) is not dict:
        epochs = {None: epochs}

    with h5py.File(filepath, "w") as hf:
        for group_name, group_epochs in epochs.items():
            grp = hf if not group_name else hf.create_group(group_name)
            for epoch, recovery_ix, session in group_epochs:
                data = epoch.drop(columns=[COL_MARKER_DEFAULT])
                dset_name = "-".join([str(int(ts)) for ts in data.index[[0, -1]]])
                subject, date, chunk = session.split(".")
                logger.debug(
                    f"Saving epoch {dset_name}: subject {subject}, date {date}, chunk {chunk}..."
                )

                dset = grp.create_dataset(
                    dset_name,
                    data=data.to_numpy(),
                    compression="gzip",
                    compression_opts=9,
                )
                dset.attrs["chunk"] = chunk
                dset.attrs["columns"] = tuple(data.columns)
                dset.attrs["date"] = date
                dset.attrs["recovery"] = recovery_ix
                dset.attrs["subject"] = subject
    logger.info("Epochs saved!")
