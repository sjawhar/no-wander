import json
import logging
from pathlib import Path
from time import gmtime, strftime
from .constants import (
    DIR_DATA_DEFAULT,
    DIR_INPUT,
    DIR_TEST,
    PREPROCESS_EXTRACT_EEG,
    PREPROCESS_NONE,
    PREPROCESS_NORMALIZE,
    SOURCE_ACC,
    SOURCE_EEG,
    SOURCE_GYRO,
    SOURCE_PPG,
)


logger = logging.getLogger(__name__)


def type_json(json_str):
    return json.loads(json_str)


def record_setup_parser(parser):
    parser.add_argument(
        "DURATION", type=int, help="Length of the meditation session in minutes",
    )
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
        "--ppg",
        dest="sources",
        action="append_const",
        const=SOURCE_PPG,
        help="Record PPG measurements",
    )
    parser.add_argument(
        "-p",
        "--probes",
        help="Intermittently sample user focus with audio probes."
        " Provide one number X to sample every X minutes."
        " Provide two numbers MEAN, STD to sample every Gaussian(MEAN, STD) minutes.",
        metavar=("MEAN", "STD"),
        nargs="+",
        type=float,
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

    run_session(get_duration(args.DURATION), args.sources, args.filename, probes=args.probes)
    end_stream()


def process_setup_parser(parser):
    parser.set_defaults(min_verbosity=1)
    parser.add_argument(
        "DATA_DIR",
        nargs="?",
        default=DIR_DATA_DEFAULT / DIR_INPUT,
        help="Directory containing data files",
    )
    parser.add_argument(
        "-l", "--limit", type=int, help="Limit the number of processed files",
    )
    parser.add_argument(
        "-s",
        "--val-split",
        type=float,
        help="Percentage of data to reserve for validation",
    )
    parser.add_argument(
        "-t",
        "--test-split",
        type=float,
        help="Percentage of data to reserve for final testing",
    )
    parser.add_argument(
        "-x",
        "--aux-channel",
        help="Channel name for Right Aux. Must be provided if Right Aux has data, otherwise channel is dropped.",
    )


def process_run(args):
    from .process import get_files_by_session, process_session_data

    if type(args.DATA_DIR) is str:
        args.DATA_DIR = Path(args.DATA_DIR)
    args.DATA_DIR = args.DATA_DIR.resolve()

    logger.debug(f"Starting command process with args {args}")

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    data_dir = kwargs.pop("DATA_DIR")

    raw_files = get_files_by_session(data_dir)
    process_session_data(raw_files, data_dir.parent, **kwargs)


def train_setup_parser(parser):
    parser.set_defaults(allow_unknown_args=True)
    parser.add_argument("DATA_FILE", help="Path to h5 file with labeled epochs")
    parser.add_argument(
        "MODEL_DIR", help="Directory in which to save built model and images"
    )
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        required=True,
        help="Number of readings/timesteps per sample",
    )
    parser.add_argument(
        "-q",
        "--sequence-size",
        type=int,
        required=True,
        help="Number of samples per LSTM sequence",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=type_json,
        required=True,
        help=" ".join(
            [
                "JSON array of layers with params. 'type' controls layer type.",
                "'ic_params' controls IC layer after activation.",
                "Include a 'pool' attribute of kwargs in a Conv1D layer to add a MaxPooling1D layer before the IC layer.",
                "A single layer controlled by '--output' is automatically added after all specified layers.",
            ]
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=type_json,
        help=" ".join(
            [
                "JSON object overriding default output layer specification.",
                "Default is a single-unit Dense layer with sigmoid activation.",
                "Do NOT include 'type' or 'name' in the layer specification.",
            ]
        ),
    )
    parser.add_argument(
        "--pre-window",
        type=float,
        nargs=2,
        help="Start and end of pre-recovery window, in seconds. Should be negative numbers (e.g. --pre-window -7 -1)",
    )
    parser.add_argument(
        "--post-window",
        type=float,
        nargs=2,
        help="Start and end of post-recovery window, in seconds.",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        choices=[PREPROCESS_EXTRACT_EEG, PREPROCESS_NONE, PREPROCESS_NORMALIZE],
        help="Type of preprocessing to perform on input data",
    )
    parser.add_argument(
        "--encode-position",
        action="store_true",
        default=None,
        help="Add positional encoding to input, before dropout",
    )
    parser.add_argument(
        "--dropout", type=float, help="Dropout rate for input",
    )
    parser.add_argument(
        "--learning-rate", type=float, help="learning_rate parameter for optimizer"
    )
    parser.add_argument(
        "--beta-one", type=float, help="beta_one parameter for optimizer"
    )
    parser.add_argument(
        "--beta-two", type=float, help="beta_two parameter for optimizer"
    )
    parser.add_argument("--decay", type=float, help="decay parameter for optimizer")
    parser.add_argument(
        "--shuffle-samples",
        action="store_true",
        default=None,
        help="Shuffle samples before constructing LSTM sequences",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("-b", "--batch-size", type=int, help="Training batch size")

    dest = "checkpoint"
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-k",
        "--checkpoint",
        action="store_true",
        default=None,
        dest=dest,
        help="Save model checkpoint every epoch",
    )
    group.add_argument("--no-checkpoint", action="store_false", dest=dest)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=None,
        help="Save TensorBoard logs every epoch",
    )
    parser.add_argument(
        "-g",
        "--gradient-metrics",
        action="store_true",
        default=None,
        help="Print metrics in Gradient chart format every epoch",
    )


def train_run(args, **kwargs):
    from .train import build_and_train_model

    kwargs.update({k: v for k, v in vars(args).items() if v is not None})
    build_and_train_model(
        kwargs.pop("DATA_FILE"),
        kwargs.pop("MODEL_DIR"),
        kwargs.pop("sample_size"),
        kwargs.pop("sequence_size"),
        kwargs.pop("layers"),
        **kwargs,
    )
