import logging
from multiprocessing import Process, Pipe, Queue
from pathlib import Path
from psychopy import core, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from .record import record_input, record_signals
from .stream import start_stream
from .constants import (
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_STREAMING_ERROR,
    EVENT_STREAMING_RESTARTED,
    EVENT_USER_RECOVER,
    PACKAGE_NAME,
)


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


def play_bell():
    try:
        bell = sound.Sound(SOUND_BELL)
        bell.play()
        core.wait(bell.getDuration())
    except:
        logger.error('Could not play bell sound')


def handle_signals_event(event, outlet):
    """Return message to pass back to signal process, else None"""
    logger.debug(f'Received {event} from signal recording')
    if event[0] == EVENT_STREAMING_ERROR:
        start_stream(None, None, confirm=False, restart=True)
        logger.info('Stream restarted')
        return [EVENT_STREAMING_RESTARTED]

    if event[0] == EVENT_RECORD_CHUNK_START:
        # muselsl.record throws an exception if marker stream contains no samples
        # Insert empty sample to ensure recording file is saved properly
        logger.debug(f'Pushing empty sample at {event[1]}')
        outlet.push_sample([MARKER_EMPTY], event[1])

    return None


def handle_input_event(event, outlet):
    """Return False to indicate recording should be ended, otherwise True"""
    logger.debug(f'Received {event} from input recording')
    if event[0] == EVENT_SESSION_END:
        outlet.push_sample([MARKER_END], event[1])
        return False

    if event[0] == EVENT_USER_RECOVER:
        outlet.push_sample([MARKER_RECOVER], event[1])

    return True


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
    play_bell()

    signals_conn, conn = Pipe()
    signals_process = Process(
        target=record_signals,
        args=(duration, sources, filepath, conn),
    )
    signals_process.start()

    input_queue = Queue()
    input_process = Process(
        target=record_input,
        args=(input_queue,),
    )
    input_process.daemon = True
    input_process.start()

    while signals_process.is_alive():
        try:
            if signals_conn.poll():
                event = handle_signals_event(signals_conn.recv(), outlet)
                if event:
                    signals_conn.send(event)
            if not input_queue.empty():
                event = input_queue.get_nowait()
                if handle_input_event(event, outlet):
                    continue
                logger.info(f'Triggering end of session at {event[1]}')
                signals_conn.send([EVENT_SESSION_END])
                signals_process.join()
                logger.info(f'Session ended at {event[1]}')
                break
        except KeyboardInterrupt:
            break

    play_bell()
    session_window.close()

    input_process.terminate()
    if signals_process.is_alive():
        signals_process.terminate()
    signals_conn.close()
    core.quit()
