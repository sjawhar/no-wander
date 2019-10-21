import argparse
from multiprocessing import Process
from muselsl import list_muses, record, stream
from os import path
from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet, IRREGULAR_RATE
from time import gmtime, sleep, strftime, time

ATTEMPTS_MAX = 5
KEYS_QUIT = ['q']
VALUE_QUIT = 2
VALUE_RESPONSE = 1

def start_stream(address):
    if address is None:
        # TODO: Debug log searching for muses
        muses = list_muses()
        if not muses or len(muses) == 0:
            # TODO: Debug log none found
            exit(1)
        # TODO: Debug log muse found
        address = muses[0]['address']

    stream_process = Process(target=stream, args=(address,))
    stream_process.daemon = True

    attempts = 0
    while attempts < ATTEMPTS_MAX:
        # TODO: Log attempts
        stream_process.start()
        stream_process.join(10)
        if stream_process.is_alive():
            return True
        attempts += 1

    return False

def record_input(duration, source_id):
    info = StreamInfo(
        name='Markers', 
        type='Markers', 
        channel_count=1, 
        nominal_srate=IRREGULAR_RATE, 
        channel_format='int32', 
        source_id=source_id,
    )
    outlet = StreamOutlet(info)

    trial_window = visual.Window(fullscr=True, color=-1)
    text = visual.TextStim(trial_window, "If you can read this, you're not meditating!", color=[1,1,1])
    text.draw()
    trial_window.flip()

    start_time = time()
    clock = core.Clock()
    clock.reset(-start_time)
    end_time = start_time + duration
    time_left = duration

    while time_left > 0:
        keys = event.waitKeys(
            maxWait=time_left,
            timeStamped=clock,
        )
        if keys is None:
            # TODO: Debug log wait expired
            break

        key, timestamp = keys[0]
        # TODO: Debug log keys
        value = VALUE_QUIT if key in KEYS_QUIT else VALUE_RESPONSE
        outlet.push_sample([value], timestamp)

        if value == VALUE_QUIT:
            # TODO: Debug log quit
            break
        time_left = end_time - clock.getTime()

    # TODO: Debug log trial complete
    trial_window.close()


def main(duration, filename, address=None):
    if not start_stream(address):
        exit(1)

    record_process = Process(target=record, args=(duration, path.abspath(filename)))
    record_process.daemon = True
    record_process.start()

    record_input(duration, filename)
    record_process.join()
    core.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture meditation session')
    parser.add_argument(
        'duration', 
        metavar='DURATION', 
        type=int, 
        help='length of the meditation session',
    )
    parser.add_argument(
        '-f', '--filename',
        default="data/" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".csv",
        help='filename where recording should be saved',
    )
    parser.add_argument(
        '-a', '--address',
        help='skip search and connect to specified address',
    )

    args = parser.parse_args()
    # TODO: Debug log args
    main(args.duration, args.filename, args.address)
