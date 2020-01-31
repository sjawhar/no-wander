import json
import logging
from pathlib import Path
from time import gmtime, strftime
from .constants import (
    DIR_DATA_DEFAULT,
    DIR_INPUT,
    DIR_TEST,
    PACKAGE_NAME,
    SOURCE_ACC,
    SOURCE_EEG,
    SOURCE_GYRO,
    SOURCE_PPG,
)


logger = logging.getLogger(PACKAGE_NAME + "." + __name__)


def type_json(json_str):
    return json.loads(json_str)


def record_setup_parser(parser):
    parser.add_argument(
        "-a", "--address", help="Skip search and stream from specified address",
    )
    parser.add_argument(
        "-c",
        "--acc",
        dest="sources",
        action="append_const",
        const=SOURCE_ACC,
        help="Record accelerometer measurements",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        required=True,
        help="Length of the meditation session in minutes",
    )
    parser.add_argument(
        "--no-eeg",
        dest="eeg",
        action="store_false",
        default=True,
        help="Don't record EEG measurements",
    )
    parser.add_argument(
        "-g",
        "--gyro",
        dest="sources",
        action="append_const",
        const=SOURCE_GYRO,
        help="Record gyroscope measurements",
    )
    parser.add_argument(
        "-p",
        "--ppg",
        dest="sources",
        action="append_const",
        const=SOURCE_PPG,
        help="Record PPG measurements",
    )
    parser.add_argument(
        "-s",
        "--skip-visualize",
        action="store_true",
        default=False,
        help="Skip visualization and stability check",
    )
    # TODO: Add subject argument
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument(
        "-f", "--filename", help="Filename for recorded data",
    )
    file_group.add_argument(
        "-t",
        "--test",
        dest="data_dir",
        action="store_const",
        const=DIR_TEST,
        default=DIR_INPUT,
        help="Store data in test directory",
    )


def record_run(args):
    from .session import get_duration, run_session
    from .stream import end_stream, start_stream, visualize

    if args.filename:
        args.filename = Path.cwd() / args.filename
    else:
        # TODO: Include subject
        args.filename = (
            DIR_DATA_DEFAULT
            / args.data_dir
            / f'{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}.csv'
        )

    if not args.sources:
        args.sources = []
    if args.eeg:
        args.sources.append(SOURCE_EEG)

    logger.debug(f"Starting command record with args {args}")

    if not start_stream(args.address, args.sources):
        exit(1)
    if not args.skip_visualize:
        visualize()

    run_session(get_duration(args.duration), args.sources, args.filename)
    end_stream()


def process_setup_parser(parser):
    parser.set_defaults(min_verbosity=1)
    parser.add_argument(
        "-d",
        "--data-dir",
        default=DIR_DATA_DEFAULT / DIR_INPUT,
        help="Directory containing data files",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of processed files",
    )
    parser.add_argument(
        "-s",
        "--test-split",
        type=float,
        default=0.2,
        help="Percentage of data to reserve for testing",
    )


def process_run(args):
    from .process import get_files_by_session, process_session_data

    if type(args.data_dir) is str:
        args.data_dir = Path.cwd() / args.data_dir

    logger.debug(f"Starting command process with args {args}")

    raw_files = get_files_by_session(args.data_dir)
    process_session_data(
        raw_files, args.data_dir.parent, limit=args.limit, test_split=args.test_split
    )


def train_setup_parser(parser):
    parser.add_argument("data_file")
    parser.add_argument("model_dir")
    parser.add_argument(
        "-s", "--sample-size", type=int, required=True,
    )
    parser.add_argument(
        "-q", "--sequence-size", type=int, required=True,
    )
    parser.add_argument(
        "-l", "--lstm", dest="lstm_layers", type=type_json, required=True,
    )
    parser.add_argument(
        "-c", "--conv1d", dest="conv1d_params", type=type_json,
    )
    parser.add_argument(
        "-d", "--dense", dest="dense_params", type=type_json,
    )
    parser.add_argument(
        "-f", "--extract-features", action="store_const", const=True, default=False,
    )
    parser.add_argument(
        "--learning-rate", type=float,
    )
    parser.add_argument(
        "--beta-one", type=float,
    )
    parser.add_argument(
        "--beta-two", type=float,
    )
    parser.add_argument(
        "--decay", type=float,
    )
    parser.add_argument(
        "--shuffle-samples", action="store_const", const=True, default=False,
    )
    parser.add_argument(
        "-t", "--test-size", type=float, default=0,
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1,
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
    )


def train_run(args):
    from .train import build_and_train_model

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    build_and_train_model(
        kwargs.pop("data_file"),
        kwargs.pop("model_dir"),
        kwargs.pop("sample_size"),
        kwargs.pop("sequence_size"),
        kwargs.pop("lstm_layers"),
        **kwargs,
    )
