import logging
from multiprocessing import Process, Pipe, Queue
from pathlib import Path
from psychopy import core, event, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from time import time
from .record import record_signals
from .stream import start_stream
from .constants import (
    DIR_ASSETS,
    EVENT_RECORD_CHUNK_START,
    EVENT_SESSION_END,
    EVENT_STREAMING_ERROR,
    EVENT_STREAMING_RESTARTED,
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


def handle_signals_message(message, outlet):
    """Return message to pass back to signal process, else None"""
    logger.debug(f"Received {message} from signal recording")
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
        outlet.push_sample([MARKER_SYNC], message[1])

    return None


def handle_keypress(keys, outlet):
    """Return True to indicate recording should be ended, otherwise False"""
    quit = False
    for key, timestamp in keys:
        logger.debug(f"{key} pressed at time {timestamp}")
        if key in KEYS_RECOVERY:
            outlet.push_sample([MARKER_USER_RECOVER], timestamp)
            continue
        elif key in KEYS_FOCUS:
            outlet.push_sample([MARKER_USER_FOCUS], timestamp)
            continue
        elif key not in KEYS_QUIT:
            continue

        logger.info(f"{key} pressed, ending user input recording")
        outlet.push_sample([MARKER_SYNC], timestamp)
        quit = True

    return quit


def run_session(duration, sources, filepath, probes=None):
    info = StreamInfo(
        name="Markers",
        type="Markers",
        channel_count=1,
        nominal_srate=IRREGULAR_RATE,
        channel_format="int32",
        source_id=filepath.name,
    )
    outlet = StreamOutlet(info)

    session_window = visual.Window(fullscr=True, color=-1)
    text = visual.TextStim(
        session_window, "If you can read this, you're not meditating!", color=[1, 1, 1],
    )
    text.draw()
    session_window.flip()
    play_sound(SOUND_SESSION_BEGIN, wait=False)

    clock = core.Clock()
    start_time = time()
    clock.reset(-start_time)
    logger.debug(f"Starting recording at {start_time}")

    signals_conn, conn = Pipe()
    signals_process = Process(
        target=record_signals, args=(duration, sources, filepath, conn),
    )
    signals_process.start()

    next_probe_time = None
    if probes is not None:
        logger.info(f"Audio probes enabled")
        probe_intervals = get_probe_intervals(duration, np.array(probes) * 60)
        probe_times = list(np.cumsum(probe_intervals) + time())[::-1]
        next_probe_time = probe_times.pop()
        logger.debug(
            f"First of {1 + len(probe_times)} audio probes at {next_probe_time}"
        )

    while signals_process.is_alive():
        try:
            if signals_conn.poll():
                message = handle_signals_message(signals_conn.recv(), outlet)
                if message is not None:
                    signals_conn.send(message)

            keys = event.getKeys(timeStamped=clock)
            if keys is not None and handle_keypress(keys, outlet) is True:
                logger.info("Triggering end of session")
                # TODO: Close session window on early end
                signals_conn.send([EVENT_SESSION_END])
                signals_process.join()
                logger.info("Session ended")
                break

            if next_probe_time is not None and time() > next_probe_time:
                marker_outlet.push_sample([MARKER_PROBE], clock.getTime())
                play_sound(SOUND_PROBE, wait=False)
                next_probe_time = None if len(probe_times) == 0 else probe_times.pop()
                logger.debug(f"Audio probe played. Next probe at {next_probe_time}")

        except KeyboardInterrupt:
            break

    play_sound(SOUND_SESSION_END)
    session_window.close()

    if signals_process.is_alive():
        signals_process.terminate()
    signals_conn.close()
    core.quit()
