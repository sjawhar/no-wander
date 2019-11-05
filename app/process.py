import logging
import pandas as pd
from .constants import DIR_FAILED, DIR_OUTPUT, DIR_PROCESSED, PACKAGE_NAME


PREFIX_MARKER = 'Marker'

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def get_raw_files(data_dir):
    raw_files = {}
    for file in data_dir.glob('*.[A-Z]*.[0-9]*.csv'):
        name_parts = file.name.split('.')
        recording = '.'.join(name_parts[:-3] + name_parts[-2:])
        if recording not in raw_files:
            raw_files[recording] = []
        raw_files[recording].append(file)

    for recording in raw_files:
        raw_files[recording].sort()

    return raw_files


def get_renamer(s, i):
    return lambda x: f'{s}_{x.replace(" ", "_")}' if x != "Marker0" else f'{PREFIX_MARKER}{i}'


def load_data_from_files(files):
    num_files = len(files)
    data = [None] * num_files
    for i in range(num_files):
        file = files[i]
        source = file.name.split('.')[-3]
        df = pd.read_csv(file, index_col=0)
        df.rename(columns=get_renamer(source, i), inplace=True)
        data[i] = df
    return data


def merge_sources(data):
    if len(data) == 1:
        return data[0]

    merged = pd.concat(data, axis=1, join='outer')
    merged = merged.apply(pd.Series.interpolate, args=('index',))
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
    merged[PREFIX_MARKER] = 0
    merged.loc[markers, PREFIX_MARKER] = 1
    return merged


def move_files(files, dest_dir):
    if not dest_dir.exists():
        dest_dir.mkdir()
    for file in files:
        file.rename(dest_dir / file.name)


def process_raw_files(raw_files, output_dir):
    if not output_dir.exists():
        output_dir.mkdir()

    for recording, files in raw_files.items():
        try:
            data = load_data_from_files(files)
            merged_df = merge_sources(data)
            merged_df.to_csv(output_dir / recording)
            move_files(files, output_dir.parent / DIR_PROCESSED)
        except Exception as e:
            logger.error(e)
            move_files(files, output_dir.parent / DIR_FAILED)
