import argparse
import logging
from multiprocessing import Process
from muselsl import list_muses, record, stream, view
from pathlib import Path
from psychopy import core, event, sound, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from time import gmtime, sleep, strftime, time


ATTEMPTS_MAX = 5
KEYS_QUIT = ['esc', 'q']
SOUND_BELL = (Path(__file__).parent / 'assets' / 'bell.wav').resolve()
VALUE_END = 2
VALUE_RESPONSE = 1

logger = logging.getLogger(__name__)


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
        while attempt < ATTEMPTS_MAX:
            attempt += 1
            logger.info(f'Beginning stream attempt {attempt}...')
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


def visualize():
    view(version=2)


def capture_input(duration, record_process, outlet):
    start_time = time()
    clock = core.Clock()
    clock.reset(-start_time)
    end_time = start_time + duration
    time_left = duration

    logger.info(f'Running session for {duration} seconds')
    logger.debug(f'Session to run from {start_time} to {end_time}')
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
        value = VALUE_END if key in KEYS_QUIT else VALUE_RESPONSE
        outlet.push_sample([value], timestamp)

        if value == VALUE_END:
            logger.info(f'{key} pressed, ending session')
            break
        time_left = end_time - clock.getTime()

    record_process.join(max(time_left, 10))
    logger.info('Session ended')


def run_session(duration, filename):
    info = StreamInfo(
        name='Markers',
        type='Markers',
        channel_count=1,
        nominal_srate=IRREGULAR_RATE,
        channel_format='int32',
        source_id=filename.name,
    )
    outlet = StreamOutlet(info)

    record_process = Process(target=record, args=(duration, filename.resolve()))
    record_process.daemon = True
    record_process.start()

    session_window = visual.Window(fullscr=True, color=-1)
    text = visual.TextStim(session_window, "If you can read this, you're not meditating!", color=[1,1,1])
    text.draw()
    session_window.flip()

    capture_input(duration, record_process, outlet)

    bell = sound.Sound(value=SOUND_BELL)
    bell.play()
    core.wait(bell.getDuration())
    session_window.close()
    core.quit()


def main(args):
    if not start_stream(args.address):
        exit(1)

    duration = get_duration(args.duration)
    visualize()
    filename = Path(__file__).parent / args.filename
    run_session(duration, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture meditation session')
    parser.add_argument(
        '-d', '--duration',
        type=int,
        help='Length of the meditation session in minutes',
    )
    parser.add_argument(
        '-f', '--filename',
        default="data/" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".csv",
        help='Filename where recording should be saved',
    )
    parser.add_argument(
        '-a', '--address',
        help='Skip search and stream from specified address',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose logging',
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(f'Received args {args}')
    main(args)
