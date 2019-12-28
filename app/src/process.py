import logging
import pandas as pd
from muselsl.constants import MUSE_SAMPLING_EEG_RATE
from pathlib import PurePath
from .constants import (
    DIR_EPOCHS,
    DIR_FAILED,
    DIR_PROCESSED,
    MARKER_RECOVER,
    PACKAGE_NAME,
    SOURCE_EEG,
)


EPOCH_SIZE_SECONDS = 10
EPOCH_SIZE_SAMPLES = MUSE_SAMPLING_EEG_RATE * EPOCH_SIZE_SECONDS
EVENT_RECOVERY = "Recovery"
PREFIX_MARKER = "Marker"

logger = logging.getLogger(PACKAGE_NAME + "." + __name__)


def get_files_by_session(data_dir):
    raw_files = {}
    for file in data_dir.glob("*.[A-Z]*.[0-9]*.csv"):
        name_parts = file.name.split(".")
        session = PurePath(".".join(name_parts[:-3] + name_parts[-2:]))
        if session not in raw_files:
            raw_files[session] = []
        raw_files[session].append(file)

    for session in raw_files:
        raw_files[session].sort()

    return raw_files


def get_renamer(s, i):
    return (
        lambda x: f'{s}_{x.replace(" ", "_")}'
        if x != "Marker0"
        else f"{PREFIX_MARKER}{i}"
    )


def load_session_data(files):
    num_files = len(files)
    data = [None] * num_files
    eeg_data = None
    for i in range(num_files):
        file = files[i]
        source = file.name.split(".")[-3]
        df = pd.read_csv(file, index_col=0)
        df.rename(columns=get_renamer(source, i), inplace=True)
        data[i] = df
        if source == SOURCE_EEG:
            eeg_data = df
    return data, eeg_data


def merge_sources(data, reindex=None):
    if len(data) == 1:
        return data[0]

    merged = pd.concat(data, axis=1, join="outer")
    merged = merged.apply(pd.Series.interpolate, args=("index",))
    if type(reindex) is pd.DataFrame:
        merged = merged.reindex(index=reindex.index)
    merged.dropna(axis=0, how="any", inplace=True)

    marker_cols = [
        col for col in merged.columns.values if col.startswith(PREFIX_MARKER)
    ]
    marker_cols.sort()
    last = 0
    markers = []
    for index in merged.index[(merged[marker_cols] == 1).any(axis=1)]:
        if index - last < 1:
            continue
        markers.append(index)
        last = index

    merged.drop(columns=marker_cols, inplace=True)

    marker_col = f"{PREFIX_MARKER}0"
    merged[marker_col] = 0
    merged.loc[markers, marker_col] = 1
    return merged


def get_epochs(merged_df):
    session_epochs = []
    data = merged_df.reset_index(drop=True)
    recoveries = data.iloc[:, -1] == 1
    if not recoveries.any():
        return session_epochs

    for recovery_ix in data.index[recoveries]:
        min_ix = max(0, recovery_ix - EPOCH_SIZE_SAMPLES)
        max_ix = recovery_ix + EPOCH_SIZE_SAMPLES + 1
        session_epochs.append(data.iloc[min_ix:max_ix])

    return session_epochs


def move_files(files, dest_dir):
    if not dest_dir.exists():
        dest_dir.mkdir()
    for file in files:
        file.rename(dest_dir / file.name)


def process_session_data(raw_files, output_dir, limit=None):
    epochs_dir = output_dir / DIR_EPOCHS
    if not epochs_dir.exists():
        epochs_dir.mkdir()

    processed_sessions = 0
    for session, files in raw_files.items():
        try:
            data, eeg_data = load_session_data(files)
            merged_df = merge_sources(data, reindex=eeg_data)
            epochs = get_epochs(merged_df)
            for epoch in range(len(epochs)):
                epochs[epoch].to_csv(
                    epochs_dir / f"{session.stem}.{epoch}{session.suffix}"
                )
            move_files(files, output_dir / DIR_PROCESSED)
        except Exception as e:
            logger.exception(e)
            move_files(files, output_dir / DIR_FAILED)

        processed_sessions += 1
        if limit is not None and processed_sessions > limit:
            break
