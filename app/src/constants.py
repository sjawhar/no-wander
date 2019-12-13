from pathlib import Path

DIR_DATA_DEFAULT = Path.cwd() / "data"
DIR_EPOCHS = "epochs"
DIR_FAILED = "failed"
DIR_INPUT = "input"
DIR_MERGED = "merged"
DIR_PROCESSED = "processed"
DIR_TEST = "test"

EVENT_RECORD_CHUNK_START = "EVENT_RECORD_CHUNK_START"
EVENT_SESSION_END = "EVENT_SESSION_END"
EVENT_STREAMING_ERROR = "EVENT_STREAMING_ERROR"
EVENT_STREAMING_RESTARTED = "EVENT_STREAMING_RESTARTED"

MARKER_RECOVER = 1
MARKER_SYNC = -1

PACKAGE_NAME = "no_wander"