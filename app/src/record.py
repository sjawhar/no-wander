import logging
from multiprocessing import Process
from muselsl import record
from pathlib import PurePath
from time import sleep, time
from .constants import (
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_STREAMING_ERROR,
    EVENT_STREAMING_RESTARTED,
    PACKAGE_NAME,
)


CHUNK_DURATION_MAX = 300
CHUNK_DURATION_MIN = 10

logger = logging.getLogger(PACKAGE_NAME + "." + __name__)


def record_signals(duration, sources, filepath, conn):
    filename_parts = filepath.name.split(".")
    # Filename format: NAME.SOURCE.CHUNK_NUM.EXT
    filename_parts = filename_parts[:-1] + [""] * 2 + filename_parts[-1:]
    start_time = time()
    get_remaining = lambda: duration + start_time - time()
    source_count = len(sources)

    chunk_num = 1
    remaining = duration
    # muselsl.record will hang if stream connection is broken.
    # Split into chunks to avoid total data loss.
    while remaining > CHUNK_DURATION_MIN:
        if conn.poll() and conn.recv() == [EVENT_SESSION_END]:
            logger.info("Session end signal received. Ending...")
            break

        chunk_duration = min(remaining, CHUNK_DURATION_MAX)
        filename_parts[-2] = str(chunk_num)
        record_processes = []
        logger.info(
            f"Starting chunk {chunk_num} at {time()} for {chunk_duration} seconds"
        )
        for source in sources:
            logger.debug(f"Starting {source} recording process...")
            filename_parts[-3] = source
            process = Process(
                target=record,
                args=(chunk_duration,),
                kwargs={
                    "filename": str(
                        PurePath(filepath.parent) / ".".join(filename_parts)
                    ),
                    "dejitter": True,
                    "data_source": source,
                },
            )
            process.start()
            record_processes.append(process)

        # Recording process will terminate early if stream is broken. Detect and restart.
        record_processes[0].join(CHUNK_DURATION_MIN)
        if not record_processes[0].is_alive():
            logger.warning("Error in stream. Restarting...")
            for process in record_processes:
                process.terminate()
            conn.send([EVENT_STREAMING_ERROR])
            # Wait for signal that stream has been restarted
            conn.recv()
            remaining = get_remaining()
            continue

        dummy_timestamp = time()
        logger.debug(f"Signaling chunk start at {dummy_timestamp}")
        conn.send([EVENT_RECORD_CHUNK_START, dummy_timestamp])
        record_processes[0].join(chunk_duration)

        # Give other recording processes a chance to end normally
        if source_count > 1:
            sleep(2)

        for si in range(source_count):
            if record_processes[si].is_alive():
                logger.warning(
                    f"{sources[si]} recording process has hung. Terminating..."
                )
                record_processes[si].terminate()

        new_remaining = get_remaining()
        chunk_time = remaining - new_remaining
        logger.debug(f"Chunk {chunk_num} took {chunk_time} seconds")
        logger.debug("%.1f seconds left in recording" % new_remaining)

        remaining = new_remaining
        chunk_num += 1

    conn.close()
