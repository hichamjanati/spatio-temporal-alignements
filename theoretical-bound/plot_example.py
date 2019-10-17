import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import euclidean_distances
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize

rc = {"legend.fontsize": 11,
      "axes.titlesize": 11,
      "axes.labelsize": 11,
      "xtick.labelsize": 9,
      "ytick.labelsize": 9,
      "pdf.fonttype": 42}
plt.rcParams.update(rc)

cmaps = ["Greens", "Oranges", "Purples"]


def gauss(size, mean, std, mass):
    """return a 1D histogram for a gaussian distribution
    (n bins, mean m and std s)
    """
    x = np.arange(size, dtype=np.float64)
    h = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    h = h[:, None] / h.sum()
    h *= mass
    return h


def gaussian_mixture(means, sds, values, gaps, T, mass=10, offset=True):
    x = []
    for m, sd in zip(means, sds):
        x.append(gauss(T, m, sd, mass))
    x = np.array(x).sum(0)

    if offset:
        starts = np.r_[- gaps[0], means]
        ends = np.r_[means, 10 * T]
        for start, end, value, gap in zip(starts, ends, values, gaps):
            x[start + gap: end - gap] = value
    return x


def generate_path(start_x, start_y, end_x, end_y, weight_x=0.33,
                  weight_y=0.77, seed=None):
    path_x = [start_x]
    path_y = [start_y]
    x, y = start_x, start_y
    rng = np.random.RandomState(seed)
    while((x, y) != (end_x, end_y)):
        if x == end_x or y == end_y:
            if x == end_x:
                y += 1
            elif y == end_y:
                x += 1
        else:
            u = rng.rand()
            if u <= weight_x:
                x += 1
            elif u >= weight_y:
                y += 1
            else:
                x += 1
                y += 1
        path_x.append(x)
        path_y.append(y)
    path_x = np.array(path_x)
    path_y = np.array(path_y)

    path = np.stack((path_x, path_y))
    return path


def make_full_path(start_links, end_links, finals, seed=None, **kwargs):
    start = (0, 0)
    paths = []
    for (start_link, end_link) in (zip(start_links, end_links)):
        path = generate_path(start[0], start[1], start_link[0], start_link[1],
                             seed=seed, **kwargs)
        link_x = np.arange(start_link[0] + 1, end_link[0])
        link_y = np.arange(start_link[1] + 1, end_link[1])
        start = end_link
        link = np.stack((link_x, link_y))
        paths.append(path)
        paths.append(link)

    path1 = generate_path(end_link[0], end_link[1], finals[0], finals[1],
                          seed=seed, **kwargs)
    paths.append(path1)
    path = np.hstack(paths)
    return path


def plot1D_mat(a, b, M, paths, title=''):
    """ Plot matrix M  with the source and target 1D distribution
    """

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    f = plt.figure(figsize=(6, 6))
    na, nb = M.shape

    gs = gridspec.GridSpec(4, 4, figure=f)

    xa = np.arange(na)
    xb = np.arange(nb)

    ax = plt.subplot(gs[1:, :-1])
    ax.imshow(M.T, interpolation='gaussian', cmap="OrRd",
              norm=Normalize(vmin=0., vmax=np.nanmax(M)))
    ax.grid(which="both", alpha=0.5)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    for path, cmap in zip(paths, cmaps):
        ax.imshow(path, cmap=cmap, vmin=0, vmax=1)

    ax1 = plt.subplot(gs[0, :-1])
    ax1.plot(xa, a, 'r', label='Target distribution')
    ax1.set_yticks([0, 5, 10, 15, 20])
    ax1.set_xticks([])

    ax1.set_title(title)
    ax1.set_xlim([0, len(b)])

    ax2 = plt.subplot(gs[1:, -1])
    ax2.plot(b, xb, 'b', label='Source distribution')
    ax2.invert_yaxis()
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_yticks(())

    ax2.invert_yaxis()
    ax2.set_ylim([len(a), 0])

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.3, hspace=0.2)
    # plt.savefig("fig/shift-example.pdf", tight_layout=True)

if __name__ == "__main__":
    T = 200
    n = np.arange(T)
    seed = 42
    k = 50
    m = 50
    sd = 4
    mass = 150
    # do gaussian unimodal
    paths = []

    m2 = m + k
    means = [100, 150]
    sds = [sd, sd]
    gap = 10
    values = [5, 0, 3]
    gaps = [6, 12, 8]
    signal = gaussian_mixture(means, sds, values, gaps, 2 * T, mass=mass,
                              offset=True)
    x = signal[50: 50 + T]
    y = signal[50 - k: 50 + T - k]
    starts = [(m - 8, m + k - 8),
              (m2 - 15, m2 + k - 15)]
    ends = [(m + 15, m + k + 15), (m2 + 8, m2 + k + 8)]
    M = euclidean_distances(x, y) ** 2
    finals = (T - 1, T - 1)
    wxs = [0.8, 0.1, 0.1]
    wys = [0.9, 0.9, 0.2]
    for i in range(3):
        wx = wxs[i]
        wy = wys[i]
        p = make_full_path(starts, ends, finals,
                           weight_x=wx, weight_y=wy, seed=seed)
        mat = np.ones_like(M) * np.nan
        mat[p[1], p[0]] = 0.8
        paths.append(mat)
    M[M < 1e-5] = np.nan
    plot1D_mat(x, y, M, paths)
    plt.savefig("fig/shift-example.pdf")
