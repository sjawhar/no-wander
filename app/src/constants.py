from pathlib import Path

COL_MARKER_PREFIX = "Marker"
COL_MARKER_DEFAULT = f"{COL_MARKER_PREFIX}0"
COL_RIGHT_AUX = "EEG_Right AUX"

DIR_DATA_DEFAULT = Path.cwd() / "data"
DIR_EPOCHS = "epochs"
DIR_FAILED = "failed"
DIR_INPUT = "input"
DIR_PROCESSED = "processed"
DIR_TEST = "test"

EVENT_RECORD_CHUNK_START = "EVENT_RECORD_CHUNK_START"
EVENT_SESSION_END = "EVENT_SESSION_END"
EVENT_STREAMING_ERROR = "EVENT_STREAMING_ERROR"
EVENT_STREAMING_RESTARTED = "EVENT_STREAMING_RESTARTED"

MARKER_RECOVER = 1
MARKER_SYNC = -1

PREPROCESS_EXTRACT_EEG = "extract-eeg"
PREPROCESS_NONE = "none"
PREPROCESS_NORMALIZE = "normalize"

SAMPLE_RATE = 256

SOURCE_ACC = "ACC"
SOURCE_EEG = "EEG"
SOURCE_GYRO = "GYRO"
SOURCE_PPG = "PPG"
