import argparse
import logging
from .constants import DIR_INPUT, DIR_TEST, PACKAGE_NAME
from .session import get_duration, run_session
from .stream import (
    end_stream,
    start_stream,
    visualize,
    SOURCE_ACC,
    SOURCE_EEG,
    SOURCE_GYRO,
    SOURCE_PPG,
)
from pathlib import Path
from time import gmtime, strftime


logger = logging.getLogger(PACKAGE_NAME)
logging.basicConfig(level=logging.WARNING)


def main(args):
    duration = get_duration(args.duration)
    filepath = (Path(__file__).parent / args.filename).resolve()

    if not start_stream(args.address, args.sources):
        exit(1)
    if not args.skip_visualize:
        visualize()

    run_session(duration, args.sources, filepath)
    end_stream()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture meditation session')
    parser.add_argument(
        '-a', '--address',
        help='Skip search and stream from specified address',
    )
    parser.add_argument(
        '-c', '--acc',
        dest='sources',
        action='append_const',
        const=SOURCE_ACC,
        help='Record accelerometer measurements',
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        required=True,
        help='Length of the meditation session in minutes',
    )
    parser.add_argument(
        '--no-eeg',
        dest='eeg',
        action='store_false',
        default=True,
        help='Record gyroscope measurements',
    )
    parser.add_argument(
        '-f', '--filename',
        help='Filename where recording should be saved',
    )
    parser.add_argument(
        '-g', '--gyro',
        dest='sources',
        action='append_const',
        const=SOURCE_GYRO,
        help='Record gyroscope measurements',
    )
    parser.add_argument(
        '-p', '--ppg',
        dest='sources',
        action='append_const',
        const=SOURCE_PPG,
        help='Record PPG measurements',
    )
    parser.add_argument(
        '-s', '--skip-visualize',
        action='store_true',
        default=False,
        help='Skip visualization and stability check',
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        default=False,
        help='Store data in test directory',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose logging',
    )

    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.filename and args.test:
        raise ValueError('Cannot use both filename and test flags at the same time.')
    elif not args.filename:
        data_dir = DIR_TEST if args.test else DIR_INPUT
        args.filename = data_dir / f'{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}.csv'

    if not args.sources:
        args.sources = []
    if args.eeg:
        args.sources.append(SOURCE_EEG)

    logger.debug(f'Received args {args}')
    main(args)
