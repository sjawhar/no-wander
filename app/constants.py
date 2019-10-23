from pathlib import Path

PACKAGE_NAME = 'meditation_eeg'
RECORD_KEYS_QUIT = ['esc', 'q']
RECORD_VALUE_END = 2
RECORD_VALUE_RESPONSE = 1
SOUND_BELL = (Path(__file__).parent / 'assets' / 'bell.wav').resolve()
STREAM_ATTEMPTS_MAX = 5
