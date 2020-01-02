import argparse
import logging
from .cli import process_run, process_setup_parser, record_run, record_setup_parser
from .constants import PACKAGE_NAME


logger = logging.getLogger(PACKAGE_NAME)
logging.basicConfig(level=logging.WARNING)


def add_verbosity_arg(parser):
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Enable verbose logging",
    )


def get_log_level(verbosity):
    if verbosity >= 2:
        return logging.DEBUG
    elif verbosity == 1:
        return logging.INFO
    return logging.WARNING


parser = argparse.ArgumentParser(prog="no_wander")
parser.set_defaults(min_verbosity=0)
add_verbosity_arg(parser)

subparsers = parser.add_subparsers(title="commands", description="valid commands")

parser_record = subparsers.add_parser("record", help="Record meditation session")
parser_record.set_defaults(handler=record_run)
add_verbosity_arg(parser_record)
record_setup_parser(parser_record)

parser_process = subparsers.add_parser(
    "process", help="Merge sources for recorded sessions"
)
parser_process.set_defaults(handler=process_run)
add_verbosity_arg(parser_process)
process_setup_parser(parser_process)

if __name__ == "__main__":
    args = parser.parse_args()

    logger.setLevel(get_log_level(args.min_verbosity + args.verbose))
    logger.debug(f"Received args {args}")
    args.handler(args)
