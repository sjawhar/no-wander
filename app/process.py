import logging
import pandas as pd
from mne import create_info, Epochs, find_events, set_log_level as mne_log_level
from mne.channels import make_standard_montage
from mne.io import RawArray
from muselsl.constants import MUSE_SAMPLING_EEG_RATE
from pathlib import PurePath
from .constants import (
    DIR_EPOCHS,
    DIR_FAILED,
    DIR_MERGED,
    DIR_PROCESSED,
    MARKER_RECOVER,
    PACKAGE_NAME,
)
from .stream import SOURCE_EEG


CHANNEL_TYPE_EEG = 'eeg'
CHANNEL_TYPE_STIM = 'stim'
EVENT_RECOVERY = 'Recovery'
EPOCHS_EVENT_ID = {EVENT_RECOVERY: MARKER_RECOVER}
MONTAGE_STANDARD_1005 = make_standard_montage('standard_1005')
PREFIX_MARKER = 'Marker'

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def get_files_by_session(data_dir):
    raw_files = {}
    for file in data_dir.glob('*.[A-Z]*.[0-9]*.csv'):
        name_parts = file.name.split('.')
        session = PurePath('.'.join(name_parts[:-3] + name_parts[-2:]))
        if session not in raw_files:
            raw_files[session] = []
        raw_files[session].append(file)

    for session in raw_files:
        raw_files[session].sort()

    return raw_files


def get_renamer(s, i):
    return lambda x: f'{s}_{x.replace(" ", "_")}' if x != "Marker0" else f'{PREFIX_MARKER}{i}'


def load_session_data(files):
    num_files = len(files)
    data = [None] * num_files
    eeg_data = None
    for i in range(num_files):
        file = files[i]
        source = file.name.split('.')[-3]
        df = pd.read_csv(file, index_col=0)
        df.rename(columns=get_renamer(source, i), inplace=True)
        data[i] = df
        if source == SOURCE_EEG:
            eeg_data = df
    return data, eeg_data


def merge_sources(data, reindex=None):
    if len(data) == 1:
        return data[0]

    merged = pd.concat(data, axis=1, join='outer')
    merged = merged.apply(pd.Series.interpolate, args=('index',))
    if type(reindex) is pd.DataFrame:
        merged = merged.reindex(index=reindex.index)
    merged.dropna(axis=0, how='any', inplace=True)

    marker_cols = [col for col in merged.columns.values if col.startswith(PREFIX_MARKER)]
    marker_cols.sort()
    last = 0
    markers = []
    for index in merged.index[(merged[marker_cols] == 1).any(axis=1)]:
        if index - last < 1:
            continue
        markers.append(index)
        last = index

    merged.drop(columns=marker_cols, inplace=True)

    marker_col = f'{PREFIX_MARKER}0'
    merged[marker_col] = 0
    merged.loc[markers, marker_col] = 1
    return merged


def get_mne_raw(data, exclude_right_aux=True):
    eeg_cols = []
    stim_col = None
    for i in range(len(data.columns)):
        col = data.columns[i]
        if PREFIX_MARKER in col:
            stim_col = col
            continue
        elif not col.startswith('EEG_'):
            continue
        elif 'AUX' in col and exclude_right_aux:
            continue
        eeg_cols.append(col)

    if len(eeg_cols) == 0:
        return None

    eeg_data = data.reindex(columns=eeg_cols + [stim_col]).values.T
    eeg_data[:-1] *= 1e-6

    ch_names = [col.replace('EEG_', '') for col in eeg_cols] + [PREFIX_MARKER]
    ch_types = [CHANNEL_TYPE_EEG] * len(eeg_cols) + [CHANNEL_TYPE_STIM]
    mne_info = create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=MUSE_SAMPLING_EEG_RATE,
        montage=MONTAGE_STANDARD_1005,
    )
    return RawArray(data=eeg_data, info=mne_info)


def get_epochs(data):
    session_epochs = []
    if not (data.iloc[:, -1] == 1).any():
        return session_epochs

    raw = get_mne_raw(data)
    if raw is None:
        return session_epochs

    raw.filter(1, 30, method='iir')
    events = find_events(raw)
    epochs = Epochs(
        raw,
        events=events,
        event_id=EPOCHS_EVENT_ID,
        tmin=-2,
        tmax=2,
        baseline=None,
        reject={'eeg': 100e-6},
        preload=True,
        verbose=False,
        picks=[CHANNEL_TYPE_EEG],
    )

    num_events = len(epochs.events)
    dropped = (1 - num_events/len(events)) * 100
    logger.info(f'{dropped:.2f}% of samples dropped')

    if num_events == 0:
        return session_epochs

    epoch_data = epochs.to_data_frame().loc[EVENT_RECOVERY]
    for epoch in epoch_data.index.levels[0]:
        session_epochs.append(epoch_data.loc[epoch])

    return session_epochs


def move_files(files, dest_dir):
    if not dest_dir.exists():
        dest_dir.mkdir()
    for file in files:
        file.rename(dest_dir / file.name)


def process_session_data(raw_files, output_dir):
    mne_log_level(logger.getEffectiveLevel())

    for dest_dir in [DIR_EPOCHS, DIR_MERGED]:
        dest_dir = output_dir / dest_dir
        if not dest_dir.exists():
            dest_dir.mkdir()

    for session, files in raw_files.items():
        try:
            data, eeg_data = load_session_data(files)
            merged_df = merge_sources(data, reindex=eeg_data)
            epochs = get_epochs(merged_df)
            for epoch in range(len(epochs)):
                epochs[epoch].to_csv(output_dir / DIR_EPOCHS / f'{session.stem}.{epoch}.{session.suffix}')
            merged_df.to_csv(output_dir / DIR_MERGED / session)
            move_files(files, output_dir / DIR_PROCESSED)
        except Exception as e:
            logger.exception(e)
            move_files(files, output_dir / DIR_FAILED)

