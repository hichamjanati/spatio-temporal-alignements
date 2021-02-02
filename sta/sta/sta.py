import numpy as np
import torch
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from .distance import SinkhornDistance
from .utils import tonumpy

from pyts.metrics.dtw import accumulated_cost_matrix, _return_results
from joblib import Parallel, delayed


def dtw(cost_mat, return_path=False):
    acc_cost_mat = accumulated_cost_matrix(cost_mat)
    dtw_dist = acc_cost_mat[-1, -1]
    res = _return_results(dtw_dist, cost_mat, acc_cost_mat,
                          False, False, return_path)
    return res


def sta_distances(x, y, metric, beta=0.01, epsilon=0.01,
                  gamma=1., return_grad=False, device="cpu",
                  return_cost_mat=False, dtype=torch.float64, **kwargs):
    """Compute STA distance matrix between spacial time series.

    Parameters:
    -----------
    x: tensor, shape (n_timestamps_1, dimension_1, dimension_2, ...)
    y: tensor, shape (n_time_series, n_timestamps_2, dimension_1, dimension_2, ...)
    metric: tensor, shape (dimension, dimension)
        OT ground kernel
    beta: float
        hyperparameter of SoftDTW

    Returns:
    --------
    sta: float or array (n_time_series,)
        distances between x and y
    """
    betas = np.asarray(beta)
    if (betas == 0).sum() and return_grad:
        raise ValueError("Stak is not differentiable with beta == 0.")
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=dtype, device=device)
        y = torch.tensor(y, dtype=dtype, device=device)
        metric = torch.tensor(metric, dtype=dtype, device=device)
        if isinstance(kwargs["amari"], np.ndarray):
            kwargs["amari"] = torch.tensor(kwargs["amari"],
                                           dtype=dtype, device=device)

    if x.ndimension() == 3:
        data_is_img = True
    else:
        data_is_img = False

    W = SinkhornDistance(x, y, metric, epsilon=epsilon,
                         gamma=gamma, return_grad=return_grad, **kwargs)
    cost_matrices = W.compute()

    if return_grad:
        w_gradients = tonumpy(W.jac)

    if cost_matrices.ndim == 2:
        cost_matrices = cost_matrices[:, None]
        if return_grad:
            w_gradients = w_gradients[:, None]
    n_timestamps_x, n_time_series, n_timestamps_y = cost_matrices.shape

    if return_grad:
        w_gradients = tonumpy(W.jac)
        gradient = np.zeros((len(betas), n_timestamps_y, n_time_series,
                             *x.shape[1:]))
        del W.jac
    sta_values = np.zeros((len(betas), n_time_series))
    for k, beta in enumerate(betas):
        for i in range(n_time_series):
            if beta:
                sdtw = SoftDTW(cost_matrices[:, i], gamma=beta)
                sta_values[k, i] = sdtw.compute()
                if return_grad:
                    E = sdtw.grad()[:, :, None]
                    if data_is_img:
                        E = E[:, :, :, None]
                    gradient[k, :, i] = (w_gradients[:, i] * E).sum(axis=1)
            else:
                sta_values[k, i] = dtw(cost_matrices[:, i])
    if len(betas) == 1:
        sta_values = sta_values[0]
        gradient = gradient[0]
    if return_grad:
        return sta_values, gradient
    if return_cost_mat:
        return sta_values
    return sta_values


def sta_distances_parallel(X, i, K, betas, epsilon, gamma, n_gpu_devices=0):
    """Computes all the sta distances STA(X[i], X[i:]) for many betas."""
    n = len(X)
    print(f"Doing {i} / {n} ...")

    # set the GPU device
    if n_gpu_devices:
        device_id = i % n_gpu_devices
        device = "cuda:%s" % device_id
        torch.cuda.set_device(device_id)
    else:
        device = "cpu"

    x = torch.tensor(X[i].copy(), dtype=torch.float64, device=device)
    y = torch.tensor(X[i:].copy(), dtype=torch.float64, device=device)
    met = torch.tensor(K.copy(), dtype=torch.float64, device=device)
    out = sta_distances(x, y, met, betas, epsilon, gamma,
                        device=device, amari=None)
    return out


def sta_matrix(X, betas, K, epsilon, gamma, n_jobs=1, n_gpu_devices=0):
    """Computes the STA distance matrix in parallel."""
    n_samples = len(X)
    pll = Parallel(n_jobs=n_jobs, backend="multiprocessing")
    samples = []
    count = 0
    for i in range(n_samples):
        samples.append(i)
        count += 1
        if count == n_samples:
            break
        samples.append(n_samples - 1 - i)
        count += 1
        if count == n_samples:
            break
    iterator = (delayed(sta_distances_parallel)(X.copy(), i, K.copy(), betas,
                                                epsilon, gamma, n_gpu_devices)
                for i in samples)
    out = pll(iterator)
    matrix = np.zeros((len(betas), n_samples, n_samples))
    for i, n in enumerate(samples):
        for k in range(len(betas)):
            matrix[k, n, n:] = out[i][k]
            matrix[k, n:, n] = out[i][k]
    return matrix


def compute_sdtw(x, y, beta, i):
    """Computes soft-dtw in parallel between a pair of time series x and y."""
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    D = SquaredEuclidean(x, y)
    if beta:
        o = SoftDTW(D, gamma=beta).compute()
    else:
        D = D.compute()
        o = dtw(D)
    print("Softdtw out of %s" % i)
    return o


def sdtw_matrix(X, beta, n_jobs=1):
    """Computes the soft-dtw distance matrix in parallel for the whole data."""
    pll = Parallel(n_jobs)
    n_samples = len(X)
    X = X.reshape(n_samples, X.shape[1], -1)
    chunks = [(X[i], X[j]) for i in range(n_samples)
              for j in range(i, n_samples)]
    iterator = (delayed(compute_sdtw)(x, y, beta, i)
                for i, (x, y) in enumerate(chunks))
    out = pll(iterator)
    out = np.array(out)

    matrix = np.zeros((n_samples, n_samples))
    start = 0
    end = n_samples
    for i in range(n_samples):
        matrix[i, i:] = out[start:end]
        start = end
        end = end + n_samples - i - 1
    matrix = 0.5 * (matrix + matrix.T)
    return matrix
