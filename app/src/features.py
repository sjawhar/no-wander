import numpy as np
from .constants import (
    PREPROCESS_EXTRACT_EEG,
    PREPROCESS_NONE,
    PREPROCESS_NORMALIZE,
    SAMPLE_RATE,
)


def get_eeg_data(X_raw, features):
    features_eeg = [col for col in features if "EEG_" in col]
    X_eeg = np.nan_to_num(X_raw[:, :, np.isin(features, features_eeg)], 0)
    # For most of our EEG feature extractors, it's easier to reason about the computation
    # if the array is (epochs, channels, time)
    X_eeg = X_eeg.swapaxes(2, 1)
    return X_eeg, [col.replace("EEG_", "") for col in features_eeg]


def get_zero_crossings(data):
    is_2d = len(data.shape) == 2
    if is_2d:
        data = np.array([data])
    positive = data > 0
    crossings = np.bitwise_xor(positive[:, :, 1:], positive[:, :, :-1])
    return crossings[0] if is_2d else crossings


def extract_time_features(X):
    from scipy import stats

    enrichers = [
        ("mean", np.mean, {}),
        ("var", np.var, {}),
        ("stddev", np.std, {}),
        ("skew", stats.skew, {}),
        ("kurtosis", stats.kurtosis, {}),
        (
            "zero_xings",
            lambda x, **kwargs: np.count_nonzero(get_zero_crossings(x), **kwargs),
            {},
        ),
        ("p2p", lambda x, **kwargs: x.max(**kwargs) - x.min(**kwargs), {}),
        (
            "aauc",
            lambda x, **kwargs: np.trapz(np.abs(x), **kwargs),
            {"dx": 1 / SAMPLE_RATE},
        ),
    ]
    X_enriched = np.concatenate(
        [enrich(X, axis=-1, **kwargs) for (_, enrich, kwargs) in enrichers], axis=-1,
    )

    return X_enriched, None, [feat for (feat, _, _) in enrichers]


def extract_frequency_features(X):
    from mne.time_frequency import psd_array_multitaper
    from pywt import wavedec

    psds, freqs = psd_array_multitaper(X, SAMPLE_RATE, fmax=110)
    bands = [
        ["delta", 0],
        ["theta", 4],
        ["alpha", 8],
        ["beta", 14],
        ["gamma1", 30],
        ["gamma2", 65],
    ]
    num_bands = len(bands)

    total = psds.sum(axis=-1)
    total_no_zeros = np.where(total > 0, total, 1)
    for i in range(num_bands):
        band = bands[i]
        _, fmin = band
        fmax = (freqs[-1] + 1) if i == (num_bands - 1) else bands[i + 1][1]
        band[1] = psds[:, :, (freqs >= fmin) & (freqs < fmax)].sum() / total_no_zeros
    bands.append(["energy", total])

    dwt_level = 7
    wavelet = "db4"
    coeff_approx, coeff_detail = wavedec(X, wavelet, level=dwt_level)[:2]
    X_enriched = np.concatenate(
        [band for (_, band) in bands]
        + [
            coeff_approx.swapaxes(2, 1).reshape(X.shape[0], -1),
            coeff_detail.swapaxes(2, 1).reshape(X.shape[0], -1),
        ],
        axis=-1,
    )

    features = [band for (band, _) in bands]
    features += [f"cA{dwt_level}_{i}" for i in range(coeff_approx.shape[-1])]
    features += [f"cD{dwt_level}_{i}" for i in range(coeff_detail.shape[-1])]

    extractor = {"wavedec": {"wavelet": wavelet, "level": dwt_level}}
    return X_enriched, extractor, features


def extract_epoch_graph_features(W):
    import bct

    L = bct.weight_conversion(W, "lengths")
    L[W == 0] = np.inf
    D, _ = bct.distance_wei(L)

    l, eff, ecc, radius, diameter = bct.charpath(D, include_infinite=False)

    return [
        bct.clustering_coef_wu(W),
        bct.efficiency_wei(W, local=True),
        bct.betweenness_wei(L),
        ecc,
        [l, eff, radius, diameter],
    ]


def extract_epoch_correlation_features(X, lag_size, num_lags):
    # Since correlation is symmetric, we only need half the lags
    # C_xy(-t) = C_yx(t)
    num_lags = num_lags // 2 + 1

    X_lagged = np.full((*X.shape, num_lags), np.nan)
    X_lagged[:, :, 0] = X - X.mean(axis=-1, keepdims=True)

    for i in range(1, num_lags):
        lag = lag_size * i
        X_lagged[:, :-lag, i] = X_lagged[:, lag:, 0]

    X_lagged = X_lagged.swapaxes(2, 1).reshape(-1, X.shape[1])
    # Time-lagged signals have different lengths
    weights = np.logical_not(np.isnan(X_lagged)).sum(axis=-1)
    weights = X.shape[1] / np.minimum(weights[:, np.newaxis], weights[np.newaxis, :])

    cov = np.nan_to_num(np.cov(X_lagged, rowvar=True) * weights, 0)
    # cov[i, t1, j, t2] represents covariance of channel i lagged by t1 with channel j lagged by t2
    cov = cov.reshape(X.shape[0], num_lags, X.shape[0], num_lags)

    channels = np.arange(X.shape[0])
    autocorr = cov[channels, 0, channels].reshape(channels.size, -1)
    decorr_t = np.argmax(get_zero_crossings(autocorr), axis=-1).astype(np.float)
    decorr_t *= lag_size / SAMPLE_RATE

    var = cov[channels, 0, channels, 0].reshape(channels.size, 1)
    denom = np.sqrt(np.matmul(var, var.T))

    corr = np.abs(cov).max(axis=(-1, 1)) / np.where(denom == 0, np.inf, denom)
    graph = extract_epoch_graph_features(corr)
    corr = corr[np.tril_indices(corr.shape[0], -1)]

    return np.concatenate([corr, decorr_t, *graph])


def extract_correlation_features(X, channels):
    lag_size = 4
    num_lags = 2 ** int(np.log2(0.5 * X.shape[2] / lag_size))

    num_epochs = X.shape[0]
    num_channels = len(channels)
    num_features = (
        +(num_channels * (num_channels - 1)) // 2  # cross-correlation
        + num_channels  # decorrelation-time
        + 4 * num_channels  # graph (local)
        + 4  # graph (global)
    )
    X_corr = np.zeros((num_epochs, num_features))

    for i in range(num_epochs):
        X_corr[i] = extract_epoch_correlation_features(X[i], lag_size, num_lags)

    features_corr = [
        f"{channels[i]}_{col2}_corr"
        for i in range(num_channels - 1)
        for col2 in channels[i + 1 :]
    ]
    features_corr += [f"{col}_decor_t" for col in channels]
    features_corr += [
        f"{col}_{feat}"
        for feat in ["clust_coef", "eff", "centrality", "ecc"]
        for col in channels
    ]
    features_corr += ["charpath", "eff", "radius", "diameter"]

    return X_corr, None, features_corr


def extract_eeg_features(X_raw, features):
    X_eeg, channels = get_eeg_data(X_raw, features)

    X_time, extractor_time, features_time = extract_time_features(X_eeg)
    X_freq, extractor_freq, features_freq = extract_frequency_features(X_eeg)
    X_corr, extractor_corr, features_corr = extract_correlation_features(
        X_eeg, channels
    )

    X_enriched = np.concatenate([X_time, X_freq, X_corr], axis=-1)
    features = features_time + features_freq
    features = [f"{channel}_{enrich}" for enrich in features for channel in channels]
    features += features_corr

    extractor = {"time": extractor_time, "freq": extractor_freq, "corr": extractor_corr}
    return X_enriched, extractor, features


def normalize_data(X_raw):
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape)
    return X, {"scaler": scaler}


def preprocess_data_train(X_raw, preprocess, features_raw):
    preprocessor = {"preprocess": preprocess, "features_raw": features_raw}

    if preprocess == PREPROCESS_EXTRACT_EEG:
        X, extractor, features = extract_eeg_features(X_raw, features_raw)
        return X, {**preprocessor, **extractor}, features, True
    elif preprocess == PREPROCESS_NONE:
        return X_raw, preprocessor, features_raw, False
    elif preprocess == PREPROCESS_NORMALIZE:
        X, scaler = normalize_data(X_raw)
        return X, {**preprocessor, **scaler}, features_raw, False
    raise ValueError(f"Unknown preprocessing type {preprocess}")


def preprocess_data_test(X_raw, preprocessor):
    preprocess = preprocessor["preprocess"]

    if preprocess == PREPROCESS_EXTRACT_EEG:
        X, _, _ = extract_eeg_features(X_raw, preprocessor["features_raw"])
        return X
    elif preprocess == PREPROCESS_NONE:
        return X_raw
    elif preprocess == PREPROCESS_NORMALIZE:
        scaler = preprocessor["scaler"]
        return scaler.transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape)
    raise ValueError(f"Unknown preprocessing type {preprocess}")
