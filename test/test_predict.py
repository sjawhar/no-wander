import numpy as np
import pytest

from no_wander import predict

CHANNEL_COUNT = 4
NUM_SEGMENTS = 5
SEGMENT_SAMPLES = 64
INPUT_SHAPE = (NUM_SEGMENTS, CHANNEL_COUNT * SEGMENT_SAMPLES)


def test_update_prediction_window_is_callable():
    assert callable(predict.update_prediction_window)


def test_update_prediction_window_initializes_when_int():
    data_segment = np.random.random((CHANNEL_COUNT, SEGMENT_SAMPLES))

    window_info = predict.update_prediction_window(NUM_SEGMENTS, data_segment)
    window_data, empty_segments = window_info

    assert type(window_data) is np.ndarray
    assert empty_segments == NUM_SEGMENTS - 1
    assert window_data.shape == (NUM_SEGMENTS, *data_segment.shape)
    assert window_data.dtype == data_segment.dtype
    assert np.all(window_data[-1] == data_segment)
    assert np.all(window_data[:-1] == 0)


def test_update_prediction_window_raises_ValueError_when_info_not_int_or_type():
    with pytest.raises(ValueError):
        predict.update_prediction_window(
            "hello", np.random.random((CHANNEL_COUNT, SEGMENT_SAMPLES))
        )


def test_update_prediction_window_empty_segments_decreases_to_zero():
    window_info = NUM_SEGMENTS
    for i in range(NUM_SEGMENTS + 1):
        data_segment = np.random.random((CHANNEL_COUNT, SEGMENT_SAMPLES))

        window_info = predict.update_prediction_window(window_info, data_segment)
        empty_segments = window_info[1]

        assert empty_segments == max(0, NUM_SEGMENTS - i - 1)


def test_update_prediction_window_throws_error_if_window_size_is_not_divisible_by_segment_size():
    window_info = predict.update_prediction_window(
        5,
        np.zeros(
            (CHANNEL_COUNT, SEGMENT_SAMPLES),
        ),
    )

    with pytest.raises(ValueError):
        predict.update_prediction_window(
            window_info, np.zeros((CHANNEL_COUNT, SEGMENT_SAMPLES - 1))
        )


def test_update_prediction_window_throws_error_if_channel_count_changes():
    window_info = predict.update_prediction_window(
        5,
        np.zeros(
            (CHANNEL_COUNT, SEGMENT_SAMPLES),
        ),
    )

    with pytest.raises(ValueError):
        predict.update_prediction_window(
            window_info, np.zeros((CHANNEL_COUNT - 1, SEGMENT_SAMPLES))
        )


def test_update_prediction_window_continues_updating_after_num_segments():
    window_info = NUM_SEGMENTS
    segments = []
    for i in range(NUM_SEGMENTS + 1):
        data_segment = np.random.random((CHANNEL_COUNT, SEGMENT_SAMPLES))
        window_info = predict.update_prediction_window(window_info, data_segment)
        segments.append(data_segment)

    data_window = window_info[0]
    assert np.all(data_window[-1] == data_segment)
    assert np.all(data_window == np.array(segments[-NUM_SEGMENTS:]))


def test_update_prediction_window_does_not_grow_beyond_num_segments():
    window_info = NUM_SEGMENTS
    for i in range(NUM_SEGMENTS + 2):
        window_info = predict.update_prediction_window(
            window_info, np.random.random((CHANNEL_COUNT, SEGMENT_SAMPLES))
        )

    data_window, empty_segments = window_info
    assert empty_segments == 0
    assert np.all(data_window.shape[0] == NUM_SEGMENTS)


def test_read_prediction_window_is_callable():
    assert callable(predict.read_prediction_window)


def test_read_prediction_window_returns_None_if_not_enough_segments():
    data_window = np.zeros((NUM_SEGMENTS, CHANNEL_COUNT, SEGMENT_SAMPLES))

    for empty_segments in range(NUM_SEGMENTS - 1, 0, -1):
        assert (
            predict.read_prediction_window((data_window, empty_segments), NUM_SEGMENTS)
            is None
        )


@pytest.mark.parametrize(
    "window_segments, input_segments, empty_segments",
    [
        (NUM_SEGMENTS, NUM_SEGMENTS, 0),
        (NUM_SEGMENTS + 1, NUM_SEGMENTS, 1),
    ],
)
def test_read_prediction_window_returns_window_if_enough_segments(
    window_segments, input_segments, empty_segments
):
    window_info = (
        np.random.random((window_segments, CHANNEL_COUNT, SEGMENT_SAMPLES)),
        empty_segments,
    )

    data_window = predict.read_prediction_window(window_info, input_segments)

    assert np.all(data_window.flatten() == window_info[0][-input_segments:].flatten())


def test_predict_distraction_is_callable():
    assert callable(predict.predict_distraction)


@pytest.mark.skip(reason="TODO")
def test_predict_distraction():
    raise NotImplementedError
