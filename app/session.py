import logging
from multiprocessing import Process, Pipe, Queue
from pathlib import Path
from psychopy import core, event, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from time import time
from .record import record_signals
from .stream import start_stream
from .constants import (
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_STREAMING_ERROR,
    EVENT_STREAMING_RESTARTED,
    PACKAGE_NAME,
)


KEYS_QUIT = ['esc', 'q']
MARKER_EMPTY = 0
MARKER_END = 2
MARKER_RECOVER = 1
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


def play_bell(wait=True):
    try:
        bell = sound.Sound(SOUND_BELL)
        bell.play()
        if not wait:
            return
        core.wait(bell.getDuration())
    except:
        logger.error('Could not play bell sound')


def handle_signals_message(message, outlet):
    """Return message to pass back to signal process, else None"""
    logger.debug(f'Received {message} from signal recording')
    if message[0] == EVENT_STREAMING_ERROR:
        start_stream(None, None, confirm=False, restart=True)
        logger.info('Stream restarted')
        return [EVENT_STREAMING_RESTARTED]

    if message[0] == EVENT_RECORD_CHUNK_START:
        # muselsl.record throws an exception if marker stream contains no samples
        # Insert empty sample to ensure recording file is saved properly
        logger.debug(f'Pushing empty sample at {message[1]}')
        outlet.push_sample([MARKER_EMPTY], message[1])

    return None


def handle_keypress(keys, outlet):
    """Return True to indicate recording should be ended, otherwise False"""
    quit = False
    for key, timestamp in keys:
        logger.debug(f'{key} pressed at time {timestamp}')
        if key not in KEYS_QUIT:
            outlet.push_sample([MARKER_RECOVER], timestamp)
            continue

        logger.info(f'{key} pressed, ending user input recording')
        outlet.push_sample([MARKER_END], timestamp)
        quit = True

    return quit


def run_session(duration, sources, filepath):
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
    play_bell(wait=False)

    clock = core.Clock()
    start_time = time()
    clock.reset(-start_time)
    logger.debug(f'Starting recording at {start_time}')

    signals_conn, conn = Pipe()
    signals_process = Process(
        target=record_signals,
        args=(duration, sources, filepath, conn),
    )
    signals_process.start()

    while signals_process.is_alive():
        try:
            if signals_conn.poll():
                message = handle_signals_message(signals_conn.recv(), outlet)
                if message is not None:
                    signals_conn.send(message)
            keys = event.getKeys(timeStamped=clock)
            if keys is not None and handle_keypress(keys, outlet) is True:
                logger.info('Triggering end of session')
                signals_conn.send([EVENT_SESSION_END])
                signals_process.join()
                logger.info('Session ended')
                break
        except KeyboardInterrupt:
            break

    play_bell()
    session_window.close()

    if signals_process.is_alive():
        signals_process.terminate()
    signals_conn.close()
    core.quit()
