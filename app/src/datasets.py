import h5py
import logging
import numpy as np
from .constants import COL_MARKER_DEFAULT, SAMPLE_RATE

logger = logging.getLogger(__name__)


def parse_dataset(filepath):
    features = {}

    with h5py.File(filepath, "r") as hf:
        data = [None] * len(hf)
        i = 0
        for dset in hf.values():
            columns = dset.attrs["columns"]
            for col in columns:
                if col not in features:
                    features[col] = len(features)
            column_numbers = [features[col] for col in columns]

            data[i] = dset[:], dset.attrs["recovery"], column_numbers
            i += 1

    return data, [feat for (feat, _) in sorted(features.items(), key=lambda x: x[1])]


def get_window_samples(data, sample_size, consecutive_samples, window):
    start, stop = window
    is_flip = stop < 0
    if is_flip:
        start, stop = -stop, -start
        data = data[::-1]

    data = data[start:stop]
    drop = len(data) % (sample_size * consecutive_samples)
    if drop > 0:
        data = data[:-drop]

    samples = data.reshape(-1, sample_size, data.shape[1])
    if is_flip:
        samples = samples[::-1, ::-1]
    return list(samples), drop


def get_samples(data, sample_size, consecutive_samples, pre_window, post_window):
    parsed = [None] * len(data)
    num_samples = 0

    for i in range(len(data)):
        dset, recovery_ix, features = data[i]
        parsed[i] = [[], [], features]
        for j, window_data, window, window_name in [
            (0, dset[:recovery_ix], pre_window, "pre"),
            (1, dset[recovery_ix:], post_window, "post"),
        ]:
            window_samples, dropped = get_window_samples(
                window_data, sample_size, consecutive_samples, window
            )
            parsed[i][j] = window_samples
            if dropped > 0:
                logger.debug(f"Dropped {dropped} readings from {window_name} window")
            num_samples += len(window_samples)

    return parsed, num_samples


def samples_to_tensors(samples, data_shape):
    X = np.full(data_shape, np.nan)
    Y = np.zeros((X.shape[0], 1))

    i = 0
    for pre, post, features in samples:
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
    sample_size,
    consecutive_samples,
    pre_window,
    post_window,
    sample_rate=SAMPLE_RATE,
):
    logger.info(f"Reading datasets from {filepath}...")
    data, features = parse_dataset(filepath)
    pre_window = tuple(int(ix * sample_rate) for ix in pre_window)
    post_window = tuple(int(ix * sample_rate) for ix in post_window)
    samples, num_samples = get_samples(
        data, sample_size, consecutive_samples, pre_window, post_window
    )
    X, Y = samples_to_tensors(samples, (num_samples, sample_size, len(features)))
    logger.debug(f"Data shape {X.shape}")
    return X, Y, features


def save_epochs(filepath, epochs):
    logger.info(f"Saving epochs to {filepath}...")
    with h5py.File(filepath, "w") as hf:
        for epoch, recovery_ix, session in epochs:
            data = epoch.drop(columns=[COL_MARKER_DEFAULT])
            dset_name = "-".join([str(int(ts)) for ts in data.index[[0, -1]]])
            date, chunk = session.split(".")[:2]
            logger.debug(f"Saving epoch {dset_name} from date {date} chunk {chunk}...")

            dset = hf.create_dataset(
                dset_name, data=data.to_numpy(), compression="gzip", compression_opts=9
            )
            dset.attrs["chunk"] = chunk
            dset.attrs["columns"] = tuple(data.columns)
            dset.attrs["date"] = date
            dset.attrs["recovery"] = recovery_ix
            # TODO: subject
            dset.attrs["subject"] = 1
    logger.info("Epochs saved!")
