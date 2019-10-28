import logging
from multiprocessing import Process
from muselsl import list_muses, record
from pathlib import PurePath
from psychopy import core, event
from time import sleep, time
from .constants import PACKAGE_NAME


CHUNK_DURATION_MAX = 300
CHUNK_DURATION_MIN = 10
KEYS_QUIT = ['esc', 'q']
VALUE_END = 2
VALUE_RESPONSE = 1

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


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
        value = VALUE_END if key in KEYS_QUIT else VALUE_RESPONSE
        outlet.push_sample([value], timestamp)

        if value == VALUE_END:
            logger.info(f'{key} pressed, ending session')
            break
        time_left = end_time - clock.getTime()

    logger.info('Session ended')


def record_chunks(duration, sources, filepath, stream_manager, outlet):
    filename_parts = filepath.name.split('.')
    filename_parts = filename_parts[:-1] + [''] * 2 + filename_parts[-1:]
    start_time = time()
    get_remaining = lambda : duration + start_time - time()

    chunk_num = 1
    remaining = duration
    while remaining > CHUNK_DURATION_MIN:
        chunk_duration = min(remaining, CHUNK_DURATION_MAX)
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
            record_processes.append(process)

        # # TODO: Push dummy sample to preserve recording if no responses
        # record_processes[0].join(5)
        # dummy_timestamp = time()
        # logger.debug(f'Pushing dummy sample at {dummy_timestamp}')
        # outlet.push_sample([VALUE_END], dummy_timestamp)

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
