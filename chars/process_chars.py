import numpy as np
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter


n_max = 5000


def process_data(n_classes=2, t_steps=21, scale=14.5, plot=False, sigma=0.):
    """Loads the raw coordinates data and produces spatio-temporal data."""
    labels = np.loadtxt("data/labels.csv", delimiter=",")[:n_max]
    i_max = np.where(labels == n_classes)[0][-1]
    mat = loadmat("data/chars.mat")
    data = list(mat["mixout"][0])[:i_max]
    n_samples = len(data)
    times = np.empty(n_samples)
    x = np.empty((n_samples, 2))
    n_pixels = np.empty(n_samples)
    dimension = np.empty(n_samples)
    integer_data = []
    for i, d in enumerate(data):
        d = d.cumsum(axis=1)
        d *= scale / abs(d[:2]).max()
        d = d.astype(int)[:2]
        times[i] = d[:, ::1].size // 2
        d += - d.min(axis=1)[:, None] + 2
        integer_data.append(d)
        n_pixels[i] = np.unique(d, axis=1).shape[1]
        dimension[i] = d.max() + 2

    dimension = int(dimension.max())
    X = np.ones((n_samples, int(times.max()), dimension, dimension)) * np.nan
    true_lengths = np.empty(n_samples).astype(int)
    for i in range(n_samples):
        true_length = integer_data[i][:, ::t_steps].shape[1]
        true_lengths[i] = true_length
        x, y = integer_data[i]
        img = np.zeros((len(x), dimension, dimension))
        img[np.arange(len(x)), x, y] = 1
        img = (img.cumsum(axis=0) > 0.)
        if sigma:
            img = gaussian_filter(img, sigma)
        X[i, :len(img)] = img

    return X[:, ::t_steps], labels[:n_samples], true_lengths


if __name__ == "__main__":
    n_classes = 8
    classes = np.arange(0, n_classes - 1)
    n_samples_per_task = 20
    t_steps = 1
    n_times = 9
    sigma = 1.

    # this scale gives 64 x 64 images
    scale = 30.5

    X, labels, _ = process_data(n_classes, t_steps=t_steps, scale=scale)
    idx_classes = np.r_[0, np.where(np.diff(labels))[0]]
    idx_classes = np.r_[0, np.where(np.diff(labels))[0]]
    ii = np.concatenate([np.arange(i + 1, i + n_samples_per_task + 1)
                         for i in idx_classes[classes]])
    X = X[ii]
    # set nan images to previous ones and apply smoothing
    for i, x in enumerate(X):
        nans = np.where(np.isnan(x))
        if len(nans[0]):
            t_final = np.where(np.isnan(x))[0][0]
            X[i, t_final:] = x[t_final - 1, None]
        for j, x in enumerate(X[i]):
            X[i, j] = gaussian_filter(x, sigma)

    y = labels[ii]
    n_times_full = X.shape[1]
    X = X[:, ::n_times_full // n_times] + 1e-3
    np.save("data/chars-processed.npy", X)
    np.save("data/chars-labels-processed.npy", y)
