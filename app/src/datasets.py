import h5py
import logging
import numpy as np
from .constants import COL_MARKER_DEFAULT, SAMPLE_RATE

WINDOW_POST_RECOVERY = (0, SAMPLE_RATE * 3)
WINDOW_PRE_RECOVERY = (SAMPLE_RATE * -7, SAMPLE_RATE * -1)

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


def get_window_samples(data, window, sample_size):
    curr, stop = window
    end = len(data)
    if curr < 0:
        curr += end
        stop += end
    while curr < 0:
        curr += sample_size

    samples = []
    while end >= curr + sample_size <= stop:
        samples.append(data[curr : curr + sample_size])
        curr += sample_size

    return samples


def get_samples(data, sample_size, include_partial_window):
    parsed = [None] * len(data)
    num_samples = 0

    for i in range(len(data)):
        dset, recovery_ix, features = data[i]
        parsed[i] = [[], [], features]
        for j, window_data, window, window_name in [
            (0, dset[:recovery_ix], WINDOW_PRE_RECOVERY, "pre"),
            (1, dset[recovery_ix:], WINDOW_POST_RECOVERY, "post"),
        ]:
            data_size = len(window_data)
            full_window_size = max(abs(p) for p in window)
            if not include_partial_window and data_size < full_window_size:
                logger.debug(f"Short {window_name} window of size {data_size} dropped")
                continue
            window_samples = get_window_samples(window_data, window, sample_size)
            parsed[i][j] = window_samples
            num_samples += len(window_samples)

    return parsed, num_samples


def samples_to_tensors(samples, data_shape):
    X = np.full(data_shape, np.nan)
    Y = np.zeros((X.shape[0], 1))

    i = 0
    for pre, post, features in samples:
        for window, label in [(pre, 0), (post, 1)]:
            i_next = i + len(window)
            if i == i_next:
                continue
            X[i:i_next, :, features] = window
            Y[i:i_next] = label
            i = i_next

    return X, Y


def read_dataset(filepath, sample_size, include_partial_window=False):
    logger.info(f"Reading datasets from {filepath}...")
    data, features = parse_dataset(filepath)
    samples, num_samples = get_samples(data, sample_size, include_partial_window)
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
