from matplotlib import pyplot as plt
import torch
import numpy as np


def groundmetric(width, height=None, p=2, normed=False):
    """Compute ground metric matrix on the 1D grid 0:`n_features`.

    Parameters
    ----------
    width : int
        width of the matrix
    height : int (optional, default equal to width)
        height of the matrix
    p: int > 0.
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: 2D array (width, height).

    """
    if height is None:
        height = width
    x = np.arange(0, width).reshape(-1, 1).astype(float)
    y = np.arange(0, height).reshape(-1, 1).astype(float)
    xx, yy = np.meshgrid(x, y)
    M = abs(xx - yy) ** p
    if normed:
        M /= np.median(M)
    return M


def groundmetric_img(width, height=None, p=2, normed=False):
    """Compute ground metric for convolutional Wasserstein.

    Parameters
    ----------
    width, height : int,
        shape of images
    p: int, optional (default 2)
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: bool (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: ndarray, shape (width, height)
    """
    if height is None:
        height = width
    M = groundmetric(width, height, p=2, normed=False)
    if normed:
        Mlarge = groundmetric2d(width, height, p=2, normed=False)
        median = np.median(Mlarge)
        M /= median
    return M


def groundmetric2d(width, height=None, p=1, normed=False):
    """Compute ground metric matrix on the 2D grid (width, height).

    Parameters
    ----------
    width: int
        The width.
    height: int | None
        The height. If None, then it defaults to width.
    p: int
        Power to raise the pairwise distance metrix. Quadratic by default.
    normed: boolean (default True)
        If True, the matrix is divided by its median.

    Returns
    -------
    M: ndarray, shape (n_features, n_features)
    """
    if height is None:
        height = width
    n_features = width * height
    M = groundmetric(width, height, p=2, normed=False)
    M = M[:, np.newaxis, :, np.newaxis] + M[np.newaxis, :, np.newaxis, :]
    M = M.reshape(n_features, n_features) ** (p / 2)

    if normed:
        M /= np.median(M)
    return M


def compute_gamma(tau, M):
    """Compute the sufficient KL weight for a full mass minimum.

    XXX add docstrings
    """
    xp = get_module(M)
    return max(0., - M.max() / (2 * xp.log(tau)))


def show_ts(X, axes=None, title="", cmap=None, show_time=False,
            normalize=True):
    """
    XXX add docstrings
    """
    T, p, p = X.shape
    vmin = X.min()
    vmax = X.max()
    if axes is None:
        f, axes = plt.subplots(T, 1)
    for i, (img, ax) in enumerate(zip(X, axes)):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_time:
            ax.set_title(f"t = {i}")
    axes[0].set_ylabel(title)


def prox_simplex_all(x):
    """Projection on the simplex.

    XXX add docstrings
    """
    shape = x.shape
    if x.ndimension() == 2:
        x = x.reshape(1, -1)
    else:
        x = x.reshape(len(x), -1)
    size = x[0].nelement()
    u = torch.sort(x, dim=1, descending=True)[0]
    cu = torch.cumsum(u, dim=1) - 1
    ucu = u - cu / torch.arange(1., size + 1., dtype=x.dtype,
                                device=x.device)[None, :]
    cucu = torch.cumsum(ucu, dim=1)
    ss = torch.argmax(cucu, dim=1)
    vv = cu[torch.arange(len(x)), ss] / (ss.type(x.dtype).to(x.device) + 1.)
    p = torch.clamp(x - vv[:, None], 0., None)
    return p.reshape(shape)


def get_module(x):
    if type(x) == np.ndarray:
        return np
    else:
        return torch


def kl(p, q):
    """Compute Kullback-Leibler divergence.

    Compute the element-wise sum of:
    `p * log(p / q) + p - q`.

    Parameters
    ----------
    p: array-like.
        must be positive.
    q: array-like.
        must be positive, same shape and dimension of `p`.

    Returns
    -------
    XXX
    """
    xp = get_module(p)
    logpq = xp.zeros_like(p)
    logpq[p > 0] = xp.log(p[p > 0] / (q[p > 0] + 1e-300))
    kl = (p * logpq + q - p)

    return kl


def wklobjective_plan(plan, p, q, K, epsilon, gamma):
    """Compute unbalanced ot objective function naively."""
    f = epsilon * kl(plan, K).sum()
    margs = kl(plan.sum(1), p) + kl(plan.sum(0), q)
    f += gamma * margs.sum()

    return f


def wklobjective(a, Kb, p, q, f0, epsilon, gamma, u=0):
    """Compute unbalanced ot objective function for solver monitoring."""
    xp = get_module(a)
    aKb = a * Kb
    f = gamma * kl(aKb, p)
    f += (aKb * (epsilon * xp.log(a + 1e-300) + u - epsilon - gamma))
    f += f0
    f += gamma * q

    return f


def tonumpy(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def fancy_imshow(matrix, ax=None):
    """imshow with values"""
    # The normal figure
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.imshow(matrix, interpolation='None',
              cmap='viridis', alpha=0.6)
    n_cols, n_rows = matrix.shape
    for j in range(n_cols):
        for i in range(n_rows):
            label = np.round(matrix.T[i, j], 2)
            ax.text(i, j, label, color='black', ha='center', va='center')
    return ax


def generate_time_series(dimension, n_times, seed=42, device="cpu",
                         dtype=torch.float64, required_grad_X=True):
    """Fake time series."""
    rng = np.random.RandomState(seed)
    x = np.zeros((n_times, dimension, dimension)) + 0.001
    y = np.zeros((n_times, dimension, dimension)) + 0.001
    ts = np.arange(n_times)
    index_x, index_y = rng.randint(dimension, size=(2, n_times))
    x[ts, index_x, index_y] = rng.rand(n_times) + 1.
    y[ts, index_y, index_x] = rng.rand(n_times) + 1.

    X = torch.tensor(x, dtype=dtype, device=device,
                     requires_grad=required_grad_X)
    Y = torch.tensor(y, dtype=dtype, device=device, requires_grad=False)
    return X, Y


def cost_matrix(dimension, device="cpu", dtype=torch.float64):
    Mimg = groundmetric(dimension, p=2, normed=False)
    M = groundmetric2d(dimension, p=2, normed=False)
    m = np.median(M)
    Mimg /= m
    M /= m
    Mimg = torch.tensor(Mimg, dtype=dtype, device=device, requires_grad=False)
    M = torch.tensor(M, dtype=dtype, device=device, requires_grad=False)

    return M, Mimg
