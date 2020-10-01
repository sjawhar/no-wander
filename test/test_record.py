import numpy as np
import pytest

from no_wander import record


CHANNEL_COUNT = 4
SEGMENT_SAMPLES = 256


def test_update_buffer_is_callable():
    assert callable(record.update_buffer)


def test_update_buffer_initializes_when_info_is_int():
    data_chunk = np.random.random((64, CHANNEL_COUNT))

    buffer_info, _ = record.update_buffer(5, data_chunk, SEGMENT_SAMPLES)
    data_buffer = buffer_info[0]

    assert type(data_buffer) is np.ndarray
    assert data_buffer.shape == (SEGMENT_SAMPLES * 5, CHANNEL_COUNT)
    assert np.all(data_buffer[:64] == data_chunk)


def test_update_buffer_raises_ValueError_when_info_is_not_int_or_tuple():
    with pytest.raises(ValueError):
        record.update_buffer(
            "hello", np.random.random((64, CHANNEL_COUNT)), SEGMENT_SAMPLES
        )


def test_update_buffer_appends_chunk_to_end():
    data_chunk = np.random.random((64, CHANNEL_COUNT))
    buffer_info = (
        np.zeros((1000, CHANNEL_COUNT)),
        17,
        0,
        SEGMENT_SAMPLES,
    )

    buffer_info = record.update_buffer(buffer_info, data_chunk, SEGMENT_SAMPLES)

    assert np.all(buffer_info[0][0][17 : 17 + 64] == data_chunk)


def test_update_buffer_throws_ValueError_on_channel_count_mismatch():
    buffer_info = record.update_buffer(
        5, np.random.random((64, CHANNEL_COUNT)), SEGMENT_SAMPLES
    )

    with pytest.raises(ValueError):
        record.update_buffer(
            buffer_info, np.random.random((64, CHANNEL_COUNT + 1)), SEGMENT_SAMPLES
        )


def test_update_buffer_returns_new_segment_when_segment_samples_reached():
    empty_samples = 15
    buffer_info = (np.zeros((1000, CHANNEL_COUNT)), 0, 0, empty_samples)
    data_chunk = np.random.random((empty_samples, CHANNEL_COUNT))

    _, is_new_segment = record.update_buffer(buffer_info, data_chunk, SEGMENT_SAMPLES)

    assert is_new_segment is True


def test_update_buffer_drops_old_data_on_overflow():
    buffer_info = (np.zeros((1000, CHANNEL_COUNT)), 975, 0, SEGMENT_SAMPLES)
    data_chunk = np.random.random((50, CHANNEL_COUNT))

    buffer_info = record.update_buffer(buffer_info, data_chunk, SEGMENT_SAMPLES)
    data_buffer = buffer_info[0][0]

    assert data_buffer.shape == (1000, CHANNEL_COUNT)
    assert np.all(data_buffer[-25:] == data_chunk[:25])
    assert np.all(data_buffer[:25] == data_chunk[25:])


def test_get_latest_segment_is_callable():
    assert callable(record.get_latest_segment)


def test_get_latest_segment_returns_zeros_for_new_buffer():
    buffer_info, _ = record.update_buffer(
        5, np.random.random((64, CHANNEL_COUNT)), SEGMENT_SAMPLES
    )

    data_segment = record.get_latest_segment(buffer_info, SEGMENT_SAMPLES)

    assert np.all(data_segment == 0)


def test_get_latest_segment_reads_contiguous_segment():
    data_chunk = np.random.random((SEGMENT_SAMPLES, CHANNEL_COUNT))
    buffer_info, _ = record.update_buffer(5, data_chunk, SEGMENT_SAMPLES)

    data_segment = record.get_latest_segment(buffer_info, SEGMENT_SAMPLES)

    assert np.all(data_segment == data_chunk)


def test_get_latest_segment_reads_wrapped_segment():
    buffer_info = (np.random.random((1000, CHANNEL_COUNT)), 0, 128, 256)

    data_segment = record.get_latest_segment(buffer_info, SEGMENT_SAMPLES)

    assert np.all(data_segment[:128] == buffer_info[0][-128:])
    assert np.all(data_segment[128:] == buffer_info[0][:128])
