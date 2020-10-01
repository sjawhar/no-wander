from multiprocessing import Process, Pipe, Queue
from pathlib import Path
from time import time
import logging
import pickle

from psychopy import core, event, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
import numpy as np

from .features import preprocess_data_test
from .predict import (
    predict_distraction,
    read_prediction_window,
    update_prediction_window,
)
from .record import monitor_signals, record_signals
from .stream import start_stream
from .constants import (
    DIR_ASSETS,
    EVENT_NEW_SEGMENT,
    EVENT_PREDICTION,
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_STREAMING_ERROR,
    EVENT_STREAMING_RESTARTED,
    MARKER_PRED_DISTRACTION,
    MARKER_PRED_FOCUS,
    MARKER_PROBE,
    MARKER_SYNC,
    MARKER_USER_FOCUS,
    MARKER_USER_RECOVER,
)

KEYS_QUIT = ["esc", "q"]
KEYS_RECOVERY = ["up", "right", "space"]
KEYS_FOCUS = ["left", "down"]
SOUND_PROBE = DIR_ASSETS / "chime.wav"
SOUND_SESSION_BEGIN = DIR_ASSETS / "bell.wav"
SOUND_SESSION_END = SOUND_SESSION_BEGIN

logger = logging.getLogger(__name__)


def get_duration(duration=None):
    while type(duration) is not int:
        try:
            duration = input("For how many minutes should the session run? ")
            duration = int(duration)
        except:
            logger.error(f"Invalid input {duration}")

    return duration * 60


def play_sound(sound_name, wait=True):
    try:
        sound_obj = sound.Sound(sound_name)
        sound_obj.play()
        if not wait:
            return
        core.wait(sound_obj.getDuration())
    except Exception as error:
        logger.error(f"Could not play {sound_name} sound: {error}")


def get_probe_intervals(duration, probes):
    try:
        # Default std of 0 if only mean is provided
        mean, std, *_ = list(probes) + [0]
    except:
        raise ValueError(f"Invalid probes configuration {probes}")

    logger.debug(f"Generating probe intervals with mean {mean} and std {std}...")
    intervals = []
    while sum(intervals) < duration:
        interval = np.random.normal(loc=mean, scale=std)
        intervals.append(mean if interval <= 0 else interval)
    intervals = intervals[:-1]

    logger.debug(f"{len(intervals)} probe intervals generated!")
    return intervals


def handle_record_message(message, marker_outlet):
    """Return message to pass back to recording process, else None"""
    logger.debug(f"Received {message} from recording process")
    if message[0] == EVENT_STREAMING_ERROR:
        # TODO: Handle case where stream never starts
        start_stream(None, None, confirm=False, restart=True)
        logger.info("Stream restarted")
        return [EVENT_STREAMING_RESTARTED]

    if message[0] == EVENT_RECORD_CHUNK_START:
        # muselsl.record throws an exception if marker stream contains no samples
        # Insert sample to ensure recording file is saved properly
        # Also helps synchronize multiple recording sources
        logger.debug(f"Pushing sync marker at {message[1]}")
        marker_outlet.push_sample([MARKER_SYNC], message[1])

    return None


def handle_keypress(keys, marker_outlet):
    """Return True to indicate recording should be ended, otherwise False"""
    quit = False
    for key, timestamp in keys:
        logger.debug(f"{key} pressed at time {timestamp}")
        if key in KEYS_RECOVERY:
            marker_outlet.push_sample([MARKER_USER_RECOVER], timestamp)
            continue
        elif key in KEYS_FOCUS:
            marker_outlet.push_sample([MARKER_USER_FOCUS], timestamp)
            continue
        elif key not in KEYS_QUIT:
            continue

        logger.info(f"{key} pressed, ending user input recording")
        marker_outlet.push_sample([MARKER_SYNC], timestamp)
        quit = True

    return quit


def get_monitor_message_handler(sources, model, preprocessor, confidence_threshold):
    source_offsets = {}
    num_channels = 0
    for source, channels in sources.items():
        source_offsets[source] = num_channels
        num_channels += len(channels)

    input_shape = model.input_shape[1:]
    window_info = input_shape[0]

    def _handle(message, marker_outlet):
        nonlocal window_info
        _, source, data_segment, timestamps = message
        # TODO: merge multiple sources
        window_info = update_prediction_window(window_info, data_segment)

        window_data = read_prediction_window(window_info, input_shape[0])
        if window_data is None:
            return None

        # TODO: Notch filter for A/C noise
        if preprocessor is not None:
            window_data = preprocess_data_test(window_data, preprocessor)

        prob = predict_distraction(model, window_data.reshape(1, *input_shape))
        is_distracted = prob > confidence_threshold

        timestamp = timestamps[-1]
        marker_outlet.push_sample(
            [MARKER_PRED_DISTRACTION if is_distracted else MARKER_PRED_FOCUS],
            timestamp,
        )
        message = [EVENT_PREDICTION, prob, is_distracted, timestamp, window_data]
        logger.debug(message)

        if not is_distracted:
            return None

    return _handle


def setup_monitoring(
    model,
    sources,
    duration,
    conn_monitor_slave,
    confidence_threshold=0.5,
    preprocessor=None,
    **kwargs,
):
    from .models import load_model

    model = load_model(model)
    if preprocessor is not None:
        with preprocessor as f:
            preprocessor = pickle.load(f)

    handle_message = get_monitor_message_handler(
        sources, model, preprocessor, confidence_threshold
    )

    monitor_process = Process(
        target=monitor_signals,
        args=(sources, duration, conn_monitor_slave),
        kwargs=kwargs,
    )

    return monitor_process, handle_message


def run_session(sources, duration, filepath, monitor=False, probes=None, **kwargs):
    session_window = visual.Window(fullscr=True, color=-1)
    text = visual.TextStim(
        session_window,
        "If you can read this, you're not meditating!",
        color=[1, 1, 1],
    )
    text.draw()
    session_window.flip()
    play_sound(SOUND_SESSION_BEGIN, wait=False)

    clock = core.Clock()
    clock.reset(-time())
    logger.debug(f"Starting recording at {clock.getTime()}")

    info = StreamInfo(
        name="Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=IRREGULAR_RATE,
        channel_format="int32",
        source_id=filepath.name,
    )
    marker_outlet = StreamOutlet(info)

    conn_record_master, conn_record_slave = Pipe()
    record_process = Process(
        target=record_signals, args=(sources, duration, conn_record_slave, filepath)
    )
    record_process.start()

    handlers = [(conn_record_master, handle_record_message)]
    monitor_process = None
    if monitor is True:
        conn_monitor_master, conn_monitor_slave = Pipe()
        monitor_process, handle_monitor_message = setup_monitoring(
            kwargs.pop("model"),
            sources,
            duration,
            conn_monitor_slave,
            **kwargs,
        )
        monitor_process.start()
        handlers.append((conn_monitor_master, handle_monitor_message))

    next_probe_time = None
    if probes is not None:
        logger.info(f"Audio probes enabled")
        probe_intervals = get_probe_intervals(duration, np.array(probes) * 60)
        probe_times = list(np.cumsum(probe_intervals) + time())[::-1]
        next_probe_time = probe_times.pop()
        logger.debug(
            f"First of {1 + len(probe_times)} audio probes at {next_probe_time}"
        )

    try:
        while record_process.is_alive():
            keys = event.getKeys(timeStamped=clock)
            if keys is not None and handle_keypress(keys, marker_outlet) is True:
                logger.info("Triggering end of session")
                conn_record_master.send([EVENT_SESSION_END])
                record_process.join()
                logger.info("Session ended")
                break

            if next_probe_time is not None and time() > next_probe_time:
                marker_outlet.push_sample([MARKER_PROBE], clock.getTime())
                play_sound(SOUND_PROBE, wait=False)
                next_probe_time = None if len(probe_times) == 0 else probe_times.pop()
                logger.debug(f"Audio probe played. Next probe at {next_probe_time}")

            for conn, handle_message in handlers:
                if not conn.poll():
                    continue

                response = handle_message(conn.recv(), marker_outlet)
                if response is None:
                    continue
                conn.send(response)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, stopping session")

    play_sound(SOUND_SESSION_END)
    session_window.close()

    if record_process.is_alive():
        record_process.terminate()

    if monitor is True:
        if monitor_process.is_alive():
            monitor_process.terminate()
        conn_monitor_slave.close()

    conn_record_master.close()
    core.quit()
