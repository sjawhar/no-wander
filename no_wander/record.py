from math import ceil
from multiprocessing import Process
from pathlib import PurePath
from time import sleep, time
import logging

from muselsl import record
from pylsl import StreamInlet, resolve_byprop
import numpy as np

from .constants import (
    EVENT_NEW_SEGMENT,
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_STREAMING_ERROR,
    EVENT_STREAMING_RESTARTED,
    SOURCE_EEG,
)

CHUNK_DURATION_BUFFER = 10
CHUNK_DURATION_MAX = 300
CHUNK_DURATION_MIN = 10

logger = logging.getLogger(__name__)


def record_signals(sources, duration, conn, filepath):
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


def pull_chunk_data(inlet, samples, channels, timeout=3, time_correction=0):
    logger.debug("Pulling data from inlet...")
    data, timestamps = inlet.pull_chunk(
        timeout=timeout * samples,
        max_samples=samples,
    )

    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, len(channels))
    data = data[:, channels]

    logger.debug(f"Pulled {data.shape[0]} samples from inlet!")
    return data, np.array(timestamps) + time_correction


# TODO: Convert to RingBuffer
def update_buffer(buffer_info, data_chunk, segment_samples):
    chunk_samples, num_channels = data_chunk.shape
    if type(buffer_info) is int:
        buffer_info = (
            np.zeros(
                (int(buffer_info * segment_samples), num_channels), dtype=np.float
            ),
            0,
            0,
            segment_samples,
        )

    buffer_info_type = type(buffer_info)
    if buffer_info_type is not tuple:
        raise ValueError(f"Invalid buffer info type: {buffer_info_type.__name__}")

    logger.debug("Adding chunk samples to buffer...")
    data_buffer, pos_data, pos_segment, empty_samples = buffer_info

    buffer_size = data_buffer.shape[0]

    pos_data_new = (pos_data + chunk_samples) % buffer_size
    logger.debug(f"New buffer position: {pos_data_new}")
    if 0 < pos_data_new < pos_data:
        logger.debug(f"Buffer filled, looping around...")
        data_buffer[pos_data:] = data_chunk[:-pos_data_new]
        data_buffer[:pos_data_new] = data_chunk[-pos_data_new:]
    else:
        pos_end = buffer_size if pos_data_new == 0 else pos_data_new
        data_buffer[pos_data:pos_end] = data_chunk

    empty_samples -= chunk_samples
    is_new_segment = empty_samples <= 0
    if is_new_segment is True:
        logger.debug("Segment filled, updating buffer info...")
        empty_samples += segment_samples
        pos_segment = (pos_segment + segment_samples) % buffer_size

    buffer_info = (data_buffer, pos_data_new, pos_segment, empty_samples)

    logger.debug("Buffer updated!")
    return buffer_info, is_new_segment


def get_latest_segment(buffer_info, segment_samples):
    logger.debug(f"Getting latest segment of size {segment_samples}...")
    data_buffer, _, pos_end, _ = buffer_info

    buffer_size = data_buffer.shape[0]
    pos_start = (pos_end - segment_samples) % buffer_size

    if pos_end > pos_start:
        return data_buffer[pos_start:pos_end]

    logger.debug("Need to loop around ring buffer to read segment")
    indices = list(range(pos_start, buffer_size)) + list(range(pos_end))
    return data_buffer[indices]


def monitor_source(
    source,
    channels,
    duration,
    conn,
    buffer_num_segments=6,
    chunk_ratio=0.25,
    chunk_timeout=3,
    segment_seconds=1,
):
    logger.info(f"Looking for an {source} stream...")
    streams = resolve_byprop("type", source, timeout=2)
    if len(streams) == 0:
        raise RuntimeError(f"Can't find {source} stream.")

    logger.info(f"{source} stream found! Monitoring stream...")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    time_correction = inlet.time_correction()

    info = inlet.info()
    sampling_rate = int(info.nominal_srate())

    segment_samples = ceil(segment_seconds * sampling_rate)
    chunk_samples = ceil(segment_samples * chunk_ratio)

    buffer_info = buffer_num_segments
    pull_timeout = segment_seconds * chunk_ratio * chunk_timeout
    start_time = time()
    try:
        while time() - start_time < duration:
            # TODO: Use source-specific chunk size from muselsl
            data_chunk, timestamps = pull_chunk_data(
                inlet,
                chunk_samples,
                channels,
                timeout=pull_timeout,
                time_correction=time_correction,
            )

            if data_chunk.shape[0] < chunk_samples:
                logger.debug(
                    f"Chunk size {data_chunk.shape[0]} less than expected {chunk_samples}"
                )
                continue

            buffer_info, is_new_segment = update_buffer(
                buffer_info, data_chunk, segment_samples
            )

            if is_new_segment is True:
                data_segment = get_latest_segment(buffer_info, segment_samples)
                logger.debug(f"New {source} segment of size {segment_samples}!")
                conn.send([EVENT_NEW_SEGMENT, source, data_segment, timestamps])

        logger.info(f"{source} monitoring ended!")

    except KeyboardInterrupt:
        logger.info(f"Closing {source} monitoring!")


def monitor_signals(
    sources,
    duration,
    *args,
    **kwargs,
):
    monitor_processes = []
    for source, channels in sources.items():
        # TODO: Add non-EEG streams
        if source != SOURCE_EEG:
            raise ValueError(f"Unsupported monitoring source {SOURCE_EEG}")

        process = Process(
            target=monitor_source,
            args=(source, channels, duration, *args),
            kwargs=kwargs,
        )
        process.start()
        monitor_processes.append(process)

    # TODO: Detect early termination and restart
    monitor_processes[0].join()
