import logging
from pathlib import Path
from time import gmtime, strftime
from .constants import DIR_INPUT, DIR_TEST, PACKAGE_NAME
from .stream import (
    SOURCE_ACC,
    SOURCE_EEG,
    SOURCE_GYRO,
    SOURCE_PPG,
)


logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def record_setup_parser(parser):
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
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument(
        '-f', '--filename',
        help='Filename for recorded data',
    )
    file_group.add_argument(
        '-t', '--test',
        action='store_true',
        default=False,
        help='Store data in test directory',
    )


def record_run(args):
    from .session import get_duration, run_session
    from .stream import end_stream, start_stream, visualize

    if not args.filename:
        data_dir = DIR_TEST if args.test else DIR_INPUT
        args.filename = data_dir / f'{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}.csv'

    if not args.sources:
        args.sources = []
    if args.eeg:
        args.sources.append(SOURCE_EEG)

    logger.debug(f'Record called with {args}')

    if not start_stream(args.address, args.sources):
        exit(1)
    if not args.skip_visualize:
        visualize()

    run_session(get_duration(args.duration), args.sources, args.filename)
    end_stream()


def process_setup_parser(parser):
    parser.add_argument(
        '-d', '--data-dir',
        default=DIR_INPUT,
        help='Directory containing data files',
    )


def process_run(args):
    from .process import get_raw_files, process_raw_files

    logger.debug(f'Process called with {args}')

    if type(args.data_dir) is str:
        args.data_dir = Path(args.data_dir)

    raw_files = get_raw_files(args.data_dir)
    process_raw_files(raw_files)
