import numpy as np
import pytest
from no_wander.datasets import get_window_samples


@pytest.fixture
def make_data():
    def _make_data(rows, cols, reverse=False):
        data = np.arange(rows * cols).reshape(rows, cols)
        if not reverse:
            return data
        return data.ravel()[::-1].reshape(data.shape)

    return _make_data


def test_get_window_samples_post_not_consecutive(make_data):
    num_features = 5
    data = make_data(10, num_features)
    sample_size = 3
    samples, dropped = get_window_samples(data, sample_size, 1, (0, 8))

    assert dropped == 2

    expected_shape = (sample_size, num_features)
    expected_sample = np.arange(sample_size * num_features).reshape(expected_shape)

    assert type(samples) is list
    assert len(samples) == 2
    for sample in samples:
        assert type(sample) is np.ndarray
        assert sample.shape == expected_shape

    assert np.all(
        samples[0] == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],]
    )
    assert np.all(samples[1] == samples[0] + samples[0].size)


def test_get_window_samples_pre_not_consecutive(make_data):
    num_features = 5
    data = make_data(10, num_features, reverse=True)
    sample_size = 3
    samples, dropped = get_window_samples(data, sample_size, 1, (-8, -1))

    assert dropped == 1

    expected_shape = (sample_size, num_features)
    expected_sample = np.arange(sample_size * num_features).reshape(expected_shape)

    assert type(samples) is list
    assert len(samples) == 2
    for sample in samples:
        assert type(sample) is np.ndarray
        assert sample.shape == expected_shape

    assert np.all(
        samples[1] == [[19, 18, 17, 16, 15], [14, 13, 12, 11, 10], [9, 8, 7, 6, 5],]
    )
    assert np.all(samples[0] == samples[1] + samples[1].size)


def test_get_window_samples_post_consecutive(make_data):
    num_features = 5
    data = make_data(15, num_features)
    sample_size = 3
    num_consecutive = 3
    samples, dropped = get_window_samples(data, sample_size, num_consecutive, (1, 14))

    assert dropped == 4

    expected_shape = (sample_size, num_features)
    expected_sample = np.arange(sample_size * num_features).reshape(expected_shape)

    assert type(samples) is list
    assert len(samples) == num_consecutive
    for sample in samples:
        assert type(sample) is np.ndarray
        assert sample.shape == expected_shape

    assert np.all(
        samples[0] == [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19],]
    )
    assert np.all(samples[1] == samples[0] + samples[0].size)


def test_get_window_samples_pre_consecutive(make_data):
    num_features = 7
    data = make_data(25, num_features, reverse=True)
    sample_size = 4
    num_consecutive = 2
    samples, dropped = get_window_samples(data, sample_size, num_consecutive, (-23, -3))

    assert dropped == 4

    expected_shape = (sample_size, num_features)
    expected_sample = np.arange(sample_size * num_features).reshape(expected_shape)

    assert type(samples) is list
    assert len(samples) == 2 * num_consecutive
    for sample in samples:
        assert type(sample) is np.ndarray
        assert sample.shape == expected_shape

    for i in range(len(samples)):
        assert np.all(
            samples[-1 - i]
            == np.array(
                [
                    [48, 47, 46, 45, 44, 43, 42],
                    [41, 40, 39, 38, 37, 36, 35],
                    [34, 33, 32, 31, 30, 29, 28],
                    [27, 26, 25, 24, 23, 22, 21],
                ]
            )
            + i * samples[-1].size
        )


@pytest.mark.parametrize(
    ("num_readings", "sample_size", "window"), [(5, 7, (1, 12)), (3, 5, (-10, -1)),]
)
def test_get_window_samples_short(num_readings, sample_size, window, make_data):
    data = make_data(num_readings, 5)
    samples, dropped = get_window_samples(data, sample_size, 1, window)

    assert dropped == num_readings - min(abs(ix) for ix in window)
    assert type(samples) is list
    assert len(samples) == 0
