import logging
from multiprocessing import Process
from muselsl import list_muses, record, stream
from pathlib import Path
from psychopy import core, event, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from time import sleep, time
from .constants import (
    PACKAGE_NAME,
    RECORD_KEYS_QUIT,
    RECORD_VALUE_END,
    RECORD_VALUE_RESPONSE,
    SOUND_BELL,
    STREAM_ATTEMPTS_MAX,
)


logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def start_stream(address):
    if address is None:
        logger.info('No address provided, searching for muses...')
        muses = list_muses()
        if not muses or len(muses) == 0:
            logger.error('No muses found, quitting.')
            exit(1)
        address = muses[0]['address']
        logger.debug(f'Found muse with address {address}.')

    keep_trying = True
    while keep_trying:
        keep_trying = False
        logger.info(f'Establishing stream to {address}...')
        attempt = 0
        while attempt < STREAM_ATTEMPTS_MAX:
            attempt += 1
            logger.info(f'Beginning stream attempt {attempt}...')
            # TODO: Option to include PPG
            # TODO: Option to include ACC
            # TODO: Option to include GYRO
            stream_process = Process(target=stream, args=(address,))
            stream_process.daemon = True
            stream_process.start()
            stream_process.join(7)
            if stream_process.is_alive():
                logger.info('Stream established!')
                return True
            logger.warning(f'Streaming attempt {attempt} failed.')

        logger.error(f'Could not establish stream after {attempt} attempts.')
        keep_trying = str.lower(input('Keep trying? (y/N) ')) in ['y', 'yes']

    return False


def get_duration(duration=None):
    while type(duration) is not int:
        try:
            duration = input('For how many minutes should the session run? ')
            duration = int(duration)
        except:
            logger.error(f'Invalid input {duration}')

    return duration * 60


def capture_input(duration, outlet, record_process):
    start_time = time()
    clock = core.Clock()
    clock.reset(-start_time)
    end_time = start_time + duration
    time_left = duration

    logger.info(f'Running session for {duration} seconds')
    logger.debug(f'Session to run from {start_time} to {end_time}')
    outlet.push_sample([0], start_time)
    while time_left > 0:
        if not record_process.is_alive():
            logger.error('Recording ended abnormally')
            break

        keys = event.waitKeys(
            maxWait=time_left,
            timeStamped=clock,
        )
        if keys is None:
            logger.debug('Session time expired')
            break

        key, timestamp = keys[0]
        logger.debug(f'{key} pressed at time {timestamp}')
        value = RECORD_VALUE_END if key in RECORD_KEYS_QUIT else RECORD_VALUE_RESPONSE
        outlet.push_sample([value], timestamp)

        if value == RECORD_VALUE_END:
            logger.info(f'{key} pressed, ending session')
            break
        time_left = end_time - clock.getTime()

    logger.info('Session ended')


def play_bell():
    try:
        bell = sound.Sound(SOUND_BELL)
        bell.play()
        core.wait(bell.getDuration())
    except:
        logger.error('Could not play bell sound')

def run_session(duration, filepath):
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
    text = visual.TextStim(session_window, "If you can read this, you're not meditating!", color=[1,1,1])
    text.draw()
    session_window.flip()
    play_bell()

    # TODO: Chunk recording into 5 minute segments
    record_process = Process(target=record, args=(duration, str(filepath.resolve())))
    record_process.start()
    capture_input(duration, outlet, record_process)

    session_window.close()
    play_bell()

    record_process.join()
    core.quit()
