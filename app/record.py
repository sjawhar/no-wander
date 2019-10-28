import logging
from multiprocessing import Process
from muselsl import list_muses, record, stream
from pathlib import Path, PurePath
from psychopy import core, event, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from time import sleep, time
from .constants import PACKAGE_NAME

RECORD_CHUNK_DURATION_MAX = 300
RECORD_CHUNK_DURATION_MIN = 10
RECORD_KEYS_QUIT = ['esc', 'q']
RECORD_VALUE_END = 2
RECORD_VALUE_RESPONSE = 1
SOUND_BELL = (Path(__file__).parent / 'assets' / 'bell.wav').resolve()
SOURCE_ACC = 'ACC'
SOURCE_EEG = 'EEG'
SOURCE_GYRO = 'GYRO'
SOURCE_PPG = 'PPG'
STREAM_ATTEMPTS_MAX = 5

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def start_stream(address, sources, confirm=True):
    if address is None:
        logger.info('No address provided, searching for muses...')
        muses = list_muses()
        if not muses or len(muses) == 0:
            logger.error('No muses found, quitting.')
            exit(1)
        address = muses[0]['address']
        logger.debug(f'Found muse with address {address}.')

    kwargs = {
        'acc_enabled': SOURCE_ACC in sources,
        'eeg_disabled': SOURCE_EEG not in sources,
        'gyro_enabled': SOURCE_GYRO in sources,
        'ppg_enabled': SOURCE_PPG in sources,
    }
    keep_trying = True
    while keep_trying:
        logger.info(f'Establishing stream to {address}...')
        attempt = 0
        while attempt < STREAM_ATTEMPTS_MAX:
            attempt += 1
            logger.info(f'Beginning stream attempt {attempt}...')
            stream_process = Process(target=stream, args=(address,), kwargs=kwargs)
            stream_process.daemon = True
            stream_process.start()
            stream_process.join(7)
            if not stream_process.is_alive():
                logger.warning(f'Streaming attempt {attempt} failed.')
                continue

            def stream_manager(restart=True):
                logger.debug(f'Stream process alive: {stream_process.is_alive()}')
                stream_process.terminate()
                return start_stream(address, sources, False) if restart else False

            logger.info('Stream established!')
            return stream_manager

        logger.error(f'Could not establish stream after {attempt} attempts.')
        if not confirm:
            continue
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


def capture_input(duration, outlet):
    start_time = time()
    clock = core.Clock()
    clock.reset(-start_time)
    end_time = start_time + duration
    time_left = duration

    logger.info(f'Running session for {duration} seconds')
    logger.debug(f'Session to run from {start_time} to {end_time}')
    while time_left > 0:
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


def record_chunks(duration, sources, filepath, stream_manager, outlet):
    filename_parts = filepath.name.split('.')
    filename_parts = filename_parts[:-1] + [''] * 2 + filename_parts[-1:]
    start_time = time()
    get_remaining = lambda : duration + start_time - time()

    chunk_num = 1
    remaining = duration
    while remaining > RECORD_CHUNK_DURATION_MIN:
        chunk_duration = min(remaining, RECORD_CHUNK_DURATION_MAX)
        filename_parts[-2] = str(chunk_num)
        record_processes = []
        logger.info(f'Starting chunk {chunk_num} at {time()} for {chunk_duration} seconds')
        for source in sources:
            logger.debug(f'Starting {source} recording process...')
            filename_parts[-3] = source
            process = Process(
                target=record,
                args=(chunk_duration,),
                kwargs={
                    'filename': str(PurePath(filepath.parent) / '.'.join(filename_parts)),
                    'dejitter': True,
                    'data_source': source,
                }
            )
            process.start()
            record_processes.push(process)

        # # TODO: Push dummy sample to preserve recording if no responses
        # record_processes[0].join(5)
        # dummy_timestamp = time()
        # logger.debug(f'Pushing dummy sample at {dummy_timestamp}')
        # outlet.push_sample([RECORD_VALUE_END], dummy_timestamp)

        record_processes[0].join(chunk_duration + 10)
        for si in range(len(sources)):
            if record_processes[si].is_alive():
                logger.warning(f'{sources[si]} recording process has idled. Terminating...')
                record_processes[si].terminate()

        chunk_time = remaining - get_remaining()
        logger.debug(f'Chunk {chunk_num} took {chunk_time} seconds')
        if chunk_time < chunk_duration:
            logger.warning('Error in stream. Restarting...')
            stream_manager = stream_manager()
            logger.info('Stream restarted')

        remaining = get_remaining()
        logger.debug('%.1f seconds left in recording' % remaining)
        chunk_num += 1


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

    record_process = Process(
        target=record_chunks,
        args=(duration, sources, filepath, stream_manager, outlet),
    )
    record_process.start()
    capture_input(duration, outlet)

    session_window.close()
    play_bell()

    record_process.join()
    core.quit()
