import logging
from multiprocessing import Process, Pipe
from pathlib import Path
from psychopy import core, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from .constants import PACKAGE_NAME
from .record import capture_input, record_chunks


SOUND_BELL = (Path(__file__).parent / 'assets' / 'bell.wav').resolve()

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def get_duration(duration=None):
    while type(duration) is not int:
        try:
            duration = input('For how many minutes should the session run? ')
            duration = int(duration)
        except:
            logger.error(f'Invalid input {duration}')

    return duration * 60


def play_bell():
    try:
        bell = sound.Sound(SOUND_BELL)
        bell.play()
        core.wait(bell.getDuration())
    except:
        logger.error('Could not play bell sound')


def run_session(duration, sources, filepath, stream_manager):
    info = StreamInfo(
        name='Markers',
        type='Markers',
        channel_count=1,
        nominal_srate=IRREGULAR_RATE,
        channel_format='int32',
        source_id=filepath.name,
    )
    outlet = StreamOutlet(info)

    session_window = visual.Window(fullscr=True, color=-1)
    text = visual.TextStim(
        session_window,
        "If you can read this, you're not meditating!",
        color=[1,1,1],
    )
    text.draw()
    session_window.flip()
    play_bell()

    record_conn, input_conn = Pipe()
    record_process = Process(
        target=record_chunks,
        args=(duration, sources, filepath, stream_manager, record_conn),
    )
    record_process.start()
    capture_input(duration, outlet, input_conn)

    session_window.close()
    play_bell()

    record_process.join()
    core.quit()
