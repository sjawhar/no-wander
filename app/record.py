import logging
from multiprocessing import Process
from muselsl import list_muses, record
from pathlib import PurePath
from psychopy import core, event
from time import sleep, time
from .constants import (
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_USER_RECOVER,
    PACKAGE_NAME,
)


CHUNK_DURATION_BUFFER = 5
CHUNK_DURATION_MAX = 300
CHUNK_DURATION_MIN = 10
KEYS_QUIT = ['esc', 'q']

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def record_input(duration, queue):
    start_time = time()
    clock = core.Clock()
    clock.reset(-start_time)
    end_time = start_time + duration
    time_left = duration

    logger.info(f'Recording user input for {duration} seconds')
    logger.debug(f'Input recording to run from {start_time} to {end_time}')
    while time_left > 0:
        keys = event.waitKeys(
            maxWait=time_left,
            timeStamped=clock,
        )
        if keys is None:
            logger.debug('User input recording time expired')
            break

        key, timestamp = keys[0]
        logger.debug(f'{key} pressed at time {timestamp}')
        if key in KEYS_QUIT:
            logger.info(f'{key} pressed, ending user input recording')
            queue.put([EVENT_SESSION_END, timestamp])
            break

        queue.put([EVENT_USER_RECOVER, timestamp])
        time_left = end_time - clock.getTime()

    queue.close()
    logger.info('User input recording ended')


def record_signals(duration, sources, filepath, stream_manager, conn):
    filename_parts = filepath.name.split('.')
    filename_parts = filename_parts[:-1] + [''] * 2 + filename_parts[-1:]
    start_time = time()
    get_remaining = lambda : duration + start_time - time()

    chunk_num = 1
    remaining = duration
    while remaining > CHUNK_DURATION_MIN:
        if conn.poll() and conn.recv() == [EVENT_SESSION_END]:
            logger.info('Session end signal received. Ending...')
            break

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

        record_processes[0].join(CHUNK_DURATION_BUFFER)
        if not record_processes[0].is_alive():
            logger.warning('Error in stream. Restarting...')
            (process.terminate() for process in record_processes)
            stream_manager = stream_manager()
            logger.info('Stream restarted')
            remaining = get_remaining()
            continue

        dummy_timestamp = time()
        logger.debug(f'Signaling chunk start at {dummy_timestamp}')
        conn.send([EVENT_RECORD_CHUNK_START, dummy_timestamp])

        record_processes[0].join(chunk_duration + CHUNK_DURATION_BUFFER)
        # Give other recording process a chance to end normally
        sleep(2)
        for si in range(len(sources)):
            if record_processes[si].is_alive():
                logger.warning(f'{sources[si]} recording process has idled. Terminating...')
                record_processes[si].terminate()

        new_remaining = get_remaining()
        chunk_time = remaining - new_remaining
        logger.debug(f'Chunk {chunk_num} took {chunk_time} seconds')
        remaining = new_remaining
        logger.debug('%.1f seconds left in recording' % remaining)
        chunk_num += 1

    conn.close()
