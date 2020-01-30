import argparse
import logging
from .cli import (
    process_run,
    process_setup_parser,
    record_run,
    record_setup_parser,
    train_run,
    train_setup_parser,
)
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

commands = [
    ("record", "Record meditation session", record_setup_parser, record_run),
    (
        "process",
        "Merge sources for recorded sessions",
        process_setup_parser,
        process_run,
    ),
    ("train", "Build and train model", train_setup_parser, train_run),
]
for command, help, setup_parser, handler in commands:
    parser_command = subparsers.add_parser(command, help=help)
    parser_command.set_defaults(command=command, handler=handler)
    add_shared_args(parser_command)
    setup_parser(parser_command)

if __name__ == "__main__":
    args = parser.parse_args()

    logger.setLevel(get_log_level(args.min_verbosity + args.verbose))
    del args.min_verbosity
    del args.verbose

    if args.debug:
        import ptvsd

        ptvsd.enable_attach(address=("0.0.0.0", 3000))
        ptvsd.wait_for_attach()
    del args.debug

    command = args.command
    handler = args.handler
    del args.command
    del args.handler

    logger.debug(f"{command} called with {args}")
    handler(args)
