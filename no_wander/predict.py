import logging

import numpy as np

logger = logging.getLogger(__name__)


def predict_distraction(model, X):
    return model(X, training=False).numpy()[0, 0, 0]


def read_prediction_window(window_info, num_segments):
    data_window, empty_segments = window_info
    if (data_window.shape[0] - empty_segments) < num_segments:
        return None

    return data_window[-num_segments:]


# TODO: Convert to RingBuffer
def update_prediction_window(window_info, data_segment):
    logger.debug(
        f"Updating prediction window with data of shape {data_segment.shape}..."
    )
    if type(window_info) is int:
        window_info = (
            np.zeros((window_info, *data_segment.shape)),
            window_info,
        )
        logger.info(f"Created prediction window of shape {window_info[0].shape}")

    window_info_type = type(window_info)
    if window_info_type is not tuple:
        raise ValueError(f"Invalid window info type: {window_info_type.__name__}")

    window_data, empty_segments = window_info
    window_data = np.roll(window_data, -1, axis=0)
    window_data[-1] = data_segment

    if empty_segments > 0:
        empty_segments -= 1

    logger.debug(f"Prediction window updated! {empty_segments} empty segments left.")
    return window_data, empty_segments
