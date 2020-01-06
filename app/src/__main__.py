import argparse
import logging
from .cli import process_run, process_setup_parser, record_run, record_setup_parser
from .constants import PACKAGE_NAME


logger = logging.getLogger(PACKAGE_NAME)
logging.basicConfig(level=logging.WARNING)


def add_shared_args(parser):
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Enable verbose logging",
    )
    parser.add_argument("--debug", action="store_true")


def get_log_level(verbosity):
    if verbosity >= 2:
        return logging.DEBUG
    elif verbosity == 1:
        return logging.INFO
    return logging.WARNING


parser = argparse.ArgumentParser(prog="no_wander")
parser.set_defaults(min_verbosity=0, debug=False)
add_shared_args(parser)

subparsers = parser.add_subparsers(title="commands", description="valid commands")

parser_record = subparsers.add_parser("record", help="Record meditation session")
parser_record.set_defaults(handler=record_run)
add_shared_args(parser_record)
record_setup_parser(parser_record)

parser_process = subparsers.add_parser(
    "process", help="Merge sources for recorded sessions"
)
parser_process.set_defaults(handler=process_run)
add_shared_args(parser_process)
process_setup_parser(parser_process)

if __name__ == "__main__":
    args = parser.parse_args()

    logger.setLevel(get_log_level(args.min_verbosity + args.verbose))
    logger.debug(f"Received args {args}")

    if args.debug:
        import ptvsd

        ptvsd.enable_attach(address=("0.0.0.0", 3000))
        ptvsd.wait_for_attach()

    args.handler(args)
