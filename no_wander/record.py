from math import ceil
from multiprocessing import Process
from pathlib import PurePath
from time import sleep, time
import logging

from muselsl import record

from .constants import (
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_STREAMING_ERROR,
    EVENT_STREAMING_RESTARTED,
)

CHUNK_DURATION_BUFFER = 10
CHUNK_DURATION_MAX = 300
CHUNK_DURATION_MIN = 10

logger = logging.getLogger(__name__)


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
        follow_process = record_processes[0]
        follow_process.join(CHUNK_DURATION_BUFFER)
        if not follow_process.is_alive():
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
        follow_process.join(chunk_duration)

        for source_idx, process in enumerate(record_processes):
            # Give recording processes a chance to end normally
            process.join(CHUNK_DURATION_BUFFER)
            if not process.is_alive():
                continue
            logger.warning(
                f"{sources[source_idx]} recording process has hung. Terminating..."
            )
            process.terminate()

        new_remaining = get_remaining()
        chunk_time = remaining - new_remaining
        logger.debug(f"Chunk {chunk_num} took {chunk_time} seconds")
        logger.debug("%.1f seconds left in recording" % new_remaining)

        remaining = new_remaining
        chunk_num += 1

    conn.close()
