import numpy as np
from matplotlib import pyplot as plt
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from joblib import delayed, Parallel
from shift_figure import gaussian_mixture


rc = {"legend.fontsize": 15,
      "axes.titlesize": 15,
      "axes.labelsize": 14,
      "xtick.labelsize": 13,
      "ytick.labelsize": 13,
      "pdf.fonttype": 42}
plt.rcParams.update(rc)


def gauss(size, mean, std, mass):
    """return a 1D histogram for a gaussian distribution (n bins, mean m and std s)
    """
    x = np.arange(size, dtype=np.float64)
    h = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    h = h[:, None] / h.sum()
    h *= mass
    return h


def compute_sdtw(signal, beta, T, k, start, end):
    x = signal[start: end]
    y = signal[start + k: end + k]
    D = SquaredEuclidean(x, y)
    o = SoftDTW(D, gamma=beta).compute()
    print("Out of %s" % k)
    return o


def psi(m, ks):
    c = 1 + 2 ** 0.5
    a = 1 - 1 / c
    fi = []
    # - a / ((ms * (c - 2) + 1))
    for k in ks:
        i = np.arange(abs(k))
        f = np.log(1 + a * i / (m - i))
        fi.append(np.sum(f))
    return np.array(fi)


def phi(m, ks):
    c = 1 + 2 ** 0.5
    a = 1 - 1 / c
    fi = []
    for k in ks:
        i = np.arange(abs(k))
        f = np.log(1 - (a * i + 1 / c) / (i + m))
        fi.append(np.sum(f))
    return np.array(fi)


def get_quadratic_bound(ks, m, mp):
    c = 1 + 2 ** 0.5
    a = 1 - 1 / c
    quadratic = a * (1 / m + 1 / (m + mp))
    quadratic *= 0.5 * (abs(ks) * (abs(ks) - 1))
    quadratic += ks / (c * (m + mp))
    return quadratic


if __name__ == "__main__":
    T_max = 2000
    T = 400
    sigma = 0.01
    betas = [0.1, 1, 50, 100]
    beta_names = ["", "1", "50", "100"]
    dists = []
    estims = []
    means = [T_max // 2 - 25, T_max // 2 + 25]
    sds = [4, 4]
    gap = 10
    values = [5, 0, 3]
    gaps = [6, 12, 8]

    mass = 150
    signal = gaussian_mixture(means, sds, values, gaps, T_max, mass=mass,
                              offset=True)
    start, end = T_max // 2 - T // 2, T_max // 2 + T // 2
    x = signal[start: end]
    m = means[0] - gaps[0] - start
    mp = end - means[-1] - gaps[-1]
    kmax = min(m, mp) - 1
    ks = np.arange(- kmax, kmax)
    for beta, beta_name in zip(betas, beta_names):
        s0 = compute_sdtw(signal, beta, T, 0, start, end)
        pll = Parallel(n_jobs=2)
        dists = pll((delayed(compute_sdtw)(signal, beta, T, k, start, end)
                     for k in ks))
        dists = np.array(dists) - s0
        dists = np.array(dists)

        t_l = - phi(m, ks)
        t_r = psi(T - mp - 1, ks)
        theoretical_bound = t_r + t_l
        theory = [t_l, t_r]
        xmin, xmax = - kmax, kmax
        ymax = None
        quadratic = get_quadratic_bound(ks, m, mp)
        quadratic *= beta
        theoretical_bound *= beta
        plt.figure()
        plt.plot(ks, dists, lw=2, marker="+", color="navy", markevery=20,
                 ms=8,
                 label=r"dtw$_\beta(x, x_{+_k})$ - dtw$_\beta(x, x)$")
        plt.plot(ks, theoretical_bound, lw=2, ms=8, color="gold",
                 marker="*",
                 markevery=20, label="Bound of (18)")
        plt.plot(ks, quadratic, color="indianred", lw=2, marker="o", ms=7,
                 markevery=20, label="Quadratic bound of (21)")
        plt.xlabel(r"Temporal shift $k$")
        plt.xlim([xmin, xmax])
        plt.ylim([0, ymax])
        plt.title(r"$\beta = %s$" % (np.round(beta, 3)))
        plt.grid()
        plt.legend()
        plt.savefig("fig/bound%s.pdf" % (beta_name))
        plt.close("all")
