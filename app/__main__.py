import argparse
import logging
from .constants import PACKAGE_NAME
from .record import get_duration, run_session, start_stream
from .viewer import visualize
from pathlib import Path
from time import gmtime, strftime


logger = logging.getLogger(PACKAGE_NAME)
logging.info(PACKAGE_NAME)


def main(args):
    duration = get_duration(args.duration)
    filepath = (Path(__file__).parent / args.filename).resolve()

    if not start_stream(args.address):
        exit(1)
    if not args.skip_visualize:
        visualize()

    run_session(duration, filepath)


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
        '-s', '--skip-visualize',
        action='store_true',
        default=False,
        help='Skip visualization and stability check',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose logging',
    )

    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.debug(f'Received args {args}')
    main(args)
