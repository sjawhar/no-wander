import logging
import numpy as np
import pandas as pd
from pathlib import PurePath
from .datasets import save_epochs
from .constants import (
    COL_MARKER_DEFAULT,
    COL_MARKER_PREFIX,
    COL_RIGHT_AUX,
    DATASET_TEST,
    DATASET_TRAIN,
    DATASET_VAL,
    DIR_EPOCHS,
    DIR_FAILED,
    DIR_PROCESSED,
    DIR_SUBJECT_PREFIX,
    MARKER_USER_RECOVER,
    SAMPLE_RATE,
    SOURCE_EEG,
)

DEBOUNCE_SECONDS = 1
EPOCH_SIZE_SECONDS = 10
EPOCH_SIZE_SAMPLES = SAMPLE_RATE * EPOCH_SIZE_SECONDS
EVENT_RECOVERY = "Recovery"
WINDOW_POST_RECOVERY = "WINDOW_POST_RECOVERY"
WINDOW_PRE_RECOVERY = "WINDOW_PRE_RECOVERY"

logger = logging.getLogger(__name__)


def get_files_by_session(
    data_dir, file_glob=f"{DIR_SUBJECT_PREFIX}*/*.[A-Z]*.[0-9]*.csv"
):
    logger.info("Collecting files for processing...")
    num_files = 0
    raw_files = {}

    for file_path in data_dir.glob(file_glob):
        subject = file_path.parent.name.replace(DIR_SUBJECT_PREFIX, "")
        datetime, _, chunk = file_path.stem.split(".")
        session = PurePath(".".join([subject, datetime, chunk]))
        if session not in raw_files:
            raw_files[session] = []
        raw_files[session].append(file_path)
        num_files += 1

    for session in raw_files:
        raw_files[session].sort()

    logger.info(
        f"Collected {num_files} files across {len(raw_files)} session chunks..."
    )
    return raw_files


def get_renamer(s, i):
    return (
        lambda x: f"{COL_MARKER_PREFIX}{i}"
        if x == COL_MARKER_DEFAULT
        else f'{s}_{x.replace(" ", "_")}'
    )


def load_session_data(files, aux_channel=None, rename=True):
    logger.debug("Loading session data...")
    data = [None] * len(files)
    eeg_data = None
    for i, datafile in enumerate(files):
        logger.debug(f"Reading {datafile} to dataframe...")
        source = datafile.name.split(".")[-3]
        df = pd.read_csv(datafile, index_col=0)
        data[i] = df

        if source == SOURCE_EEG:
            eeg_data = df

        if COL_RIGHT_AUX in df.columns:
            if aux_channel:
                logger.debug(f"Renaming {COL_RIGHT_AUX} to {aux_channel}...")
                df.rename(columns={COL_RIGHT_AUX: aux_channel}, inplace=True)
            elif not df[COL_RIGHT_AUX].any():
                logger.debug(
                    f"{COL_RIGHT_AUX} has no values and no rename provided. Dropping..."
                )
                df.drop(columns=[COL_RIGHT_AUX], inplace=True)
            else:
                raise Error(f"{COL_RIGHT_AUX} has values, must provide channel")

        if rename is True:
            df.rename(columns=get_renamer(source, i), inplace=True)

    return data, eeg_data


def merge_sources(data, reindex=None):
    logger.debug("Merging multiple data sources...")
    if len(data) == 1:
        logger.debug("Only one data source. No merge needed. Skipping...")
        return data[0]

    merged = pd.concat(data, axis=1, join="outer")
    marker_cols = [
        col for col in sorted(merged.columns) if col.startswith(COL_MARKER_PREFIX)
    ]
    merged[marker_cols] = merged[marker_cols].fillna(0)
    merged = merged.apply(pd.Series.interpolate, args=("index",))
    if type(reindex) is pd.DataFrame:
        logger.debug("Reindexing merged data sources...")
        merged = merged.reindex(index=reindex.index)
    merged.dropna(axis=0, how="any", inplace=True)

    last = 0
    markers = []
    for index in merged.index[(merged[marker_cols] == MARKER_USER_RECOVER).any(axis=1)]:
        if index - last < DEBOUNCE_SECONDS:
            logger.debug(
                f"Debouncing recovery {index}, within {DEBOUNCE_SECONDS} sec of {last}"
            )
            continue
        markers.append(index)
        last = index

    logger.debug("Consolidating marker columns...")
    merged.drop(columns=marker_cols, inplace=True)
    merged[COL_MARKER_DEFAULT] = 0
    merged.loc[markers, COL_MARKER_DEFAULT] = 1

    return merged


def get_session_epochs(merged_df, session):
    logger.debug("Splitting merged data into epochs...")
    session_epochs = []
    recoveries = merged_df.iloc[:, -1] == 1
    if not recoveries.any():
        logger.debug("No recoveries, skipping...")
        return session_epochs

    num_readings = merged_df.shape[0]
    recoveries = np.arange(num_readings)[recoveries]
    num_recoveries = len(recoveries)
    last_max = 0
    for i in range(num_recoveries):
        logger.debug(f"Extracting epoch {i + 1} of {num_recoveries}...")
        recovery_ix = recoveries[i]
        min_ix = max(last_max, recovery_ix - EPOCH_SIZE_SAMPLES)
        max_ix = min(num_readings, recovery_ix + EPOCH_SIZE_SAMPLES)

        if i < num_recoveries - 1:
            next_pre = recoveries[i + 1] - EPOCH_SIZE_SAMPLES
            if next_pre < recovery_ix + 0.5 * EPOCH_SIZE_SAMPLES:
                logger.debug("Recovery point is too close to next epoch. Discarding...")
                last_max = recovery_ix
                continue
            elif next_pre < max_ix:
                logger.debug("Post-recovery window overlaps next epoch. Shortening...")
                max_ix = next_pre

        session_epochs.append(
            (merged_df.iloc[min_ix:max_ix], recovery_ix - min_ix, session)
        )
        last_max = max_ix
        logger.debug(f"Epoch {i + 1} has {max_ix - min_ix} timestamps")

    return session_epochs


# TODO: Support splitting by subject or session
def split_epochs(epochs, val_split, test_split):
    num_epochs = len(epochs)
    test_ix = int((1 - test_split) * num_epochs)
    val_ix = int((1 - test_split - val_split) * num_epochs)
    logger.info(
        f"Splitting epochs: {val_ix} train/{test_ix - val_ix} val/{num_epochs - test_ix} test"
    )
    shuffled = np.random.permutation(epochs)
    return (
        shuffled[:val_ix],
        shuffled[val_ix:test_ix],
        shuffled[test_ix:],
    )


def move_files(files, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    created_dirs = set()
    for file in files:
        # Preserve subject subdirectory
        file_dest = dest_dir / file.parent.name
        if file_dest not in created_dirs:
            file_dest.mkdir(exist_ok=True)
            created_dirs.add(file_dest)
        file.rename(file_dest / file.name)


def process_session_data(
    raw_files, output_dir, aux_channel=None, limit=None, test_split=0.2, val_split=0.2,
):
    if not len(raw_files):
        return

    epochs_dir = output_dir / DIR_EPOCHS
    if not epochs_dir.exists():
        logger.debug(f"{epochs_dir} does not exist. Creating...")
        epochs_dir.mkdir(parents=True)

    epochs = []
    processed_files = []
    ts_range = [float("inf"), 0]
    num_chunks = len(raw_files)
    processed_chunks = 0
    logger.info(f"{num_chunks} session chunks to process. Starting...")
    for session, files in raw_files.items():
        logger.info(f"Processing session chunk {processed_chunks + 1}...")
        logger.debug(f"Session chunk name: {session}...")
        try:
            data, eeg_data = load_session_data(files, aux_channel=aux_channel)
            merged_df = merge_sources(data, reindex=eeg_data)
            session_epochs = get_session_epochs(merged_df, session.name)
            if len(session_epochs):
                epochs += session_epochs
                ts_range[0] = min(ts_range[0], merged_df.index[0])
                ts_range[1] = max(ts_range[1], merged_df.index[-1])
            processed_files += files
        except Exception as e:
            logger.exception(e)
            move_files(files, output_dir / DIR_FAILED)

        processed_chunks += 1
        logger.debug(f"Finished processing session chunk {processed_chunks}!")
        if limit is not None and processed_chunks >= limit:
            logger.info(f"Limit of {limit} reached. Breaking...")
            break

    logger.info(f"Collected {len(epochs)} epochs across {processed_chunks} chunks!")
    try:
        dataset_name = "-".join([str(int(ts)) for ts in ts_range])
        epochs_train, epochs_val, epochs_test = split_epochs(
            epochs, val_split, test_split
        )
        for dataset, split in [
            ({DATASET_TRAIN: epochs_train, DATASET_VAL: epochs_val}, DATASET_TRAIN),
            (epochs_test, DATASET_TEST),
        ]:
            save_epochs(output_dir / DIR_EPOCHS / f"{dataset_name}-{split}.h5", dataset)
        move_files(processed_files, output_dir / DIR_PROCESSED)
    except Exception as e:
        logger.exception(e)
        move_files(processed_files, output_dir / DIR_FAILED)

    logger.info("Done!")
