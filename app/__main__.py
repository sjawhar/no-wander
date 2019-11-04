import argparse
import logging
from .cli import process_run, process_setup_parser, record_run, record_setup_parser
from .constants import PACKAGE_NAME


logger = logging.getLogger(PACKAGE_NAME)
logging.basicConfig(level=logging.WARNING)


def add_verbosity_arg(parser):
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose logging',
    )

parser = argparse.ArgumentParser(prog='meditation_eeg')
add_verbosity_arg(parser)

subparsers = parser.add_subparsers(title='commands', description='valid commands')

parser_record = subparsers.add_parser('record', help='Record meditation session')
parser_record.set_defaults(handler=record_run)
add_verbosity_arg(parser_record)
record_setup_parser(parser_record)

parser_process = subparsers.add_parser('process', help='Merge sources for recorded sessions')
parser_process.set_defaults(handler=process_run)
add_verbosity_arg(parser_process)
process_setup_parser(parser_process)

if __name__ == '__main__':
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logger.debug(f'Received args {args}')
    args.handler(args)
