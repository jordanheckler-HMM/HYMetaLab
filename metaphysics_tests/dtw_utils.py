from math import inf

import numpy as np

try:
    from dtaidistance import dtw

    HAVE_DTAI = True
except Exception:
    HAVE_DTAI = False


def dtw_distance(a, b, window=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    m = len(b)
    if HAVE_DTAI:
        return float(dtw.distance(a, b))
    # classic DP
    w = window if window is not None else max(n, m)
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m + 1, i + w + 1)):
            cost = abs(a[i - 1] - b[j - 1])
            D[i][j] = cost + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    return float(D[n][m])


def dtw_matrix(series_list, window=None):
    N = len(series_list)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            d = dtw_distance(series_list[i], series_list[j], window=window)
            M[i, j] = d
            M[j, i] = d
    return M


def pca1_reduce(epoch_df, cols):
    from sklearn.decomposition import PCA

    X = epoch_df[cols].astype(float).values
    p = PCA(n_components=1)
    comp = p.fit_transform(X)[:, 0]
    return comp
