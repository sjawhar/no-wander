import h5py
import logging
import numpy as np
from .constants import COL_MARKER_DEFAULT, PACKAGE_NAME, SAMPLE_RATE

WINDOW_POST_RECOVERY = (0, SAMPLE_RATE * 3)
WINDOW_PRE_RECOVERY = (SAMPLE_RATE * -7, SAMPLE_RATE * -1)

logger = logging.getLogger(PACKAGE_NAME + "." + __name__)


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


def get_samples(data, sample_size):
    parsed = [None] * len(data)
    num_samples = 0

    for i in range(len(data)):
        dset, recovery_ix, features = data[i]
        pre = get_window_samples(dset[:recovery_ix], WINDOW_PRE_RECOVERY, sample_size)
        post = get_window_samples(dset[recovery_ix:], WINDOW_POST_RECOVERY, sample_size)
        num_samples += len(pre) + len(post)
        parsed[i] = pre, post, features

    return parsed, num_samples


def samples_to_tensors(samples, data_shape):
    num_samples, sample_size, num_features = data_shape
    X = np.full((num_samples, sample_size, num_features), np.nan)
    Y = np.zeros((num_samples, 1))

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


def read_dataset(filepath, sample_size):
    logger.info(f"Reading datasets from {filepath}...")
    data, features = parse_dataset(filepath)
    samples, num_samples = get_samples(data, sample_size)
    X, Y = samples_to_tensors(samples, (num_samples, sample_size, len(features)))
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
