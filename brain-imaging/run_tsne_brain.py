
import mne
import pickle

import numpy as np
from sta import sta_matrix, sdtw_matrix
from sklearn.manifold import TSNE

# change this if you have GPUs
# in our platform, this experiment ran on 4 GPUs in around 20 minutes

n_gpu_devices = 0


def generate_samples(n_samples, n_times, time_point, space_points, M,
                     smoothing_time=1., smoothing_space=0.01,
                     seed=None):
    """Simulate brain signals at a time_point and in a random vertex among
       `space_points`."""
    rng = np.random.RandomState(seed)
    n_features = len(M)
    time_points = (np.ones(n_samples) * time_point).astype(int)
    space_points = rng.choice(space_points, size=n_samples)
    signals = np.zeros((n_samples, n_times, n_features)).astype(float)
    values = rng.rand(n_samples) * 2 + 1
    signals[np.arange(n_samples), time_points, space_points] = values

    # create temporal and spatial gaussian filters to smooth the data
    times = np.arange(n_times)
    metric = (times[:, None] - times[None, :]) ** 2
    kernel_time = np.exp(- metric / smoothing_time)
    kernel_space = np.exp(- M / smoothing_space)

    for i, signal in enumerate(signals):
        signals[i] = kernel_space.dot(signal.T).T
        signals[i] = kernel_time.dot(signal)
    return signals

if __name__ == "__main__":

    # load brain regions
    mt = mne.read_label("data/lh.MT.label")
    v1 = mne.read_label("data/lh.V1.label")

    # load ground metric defined on the cortical triangulated mesh
    M_ = np.load("data/ground_metric.npy") ** 2
    M = M_ / np.median(M_)

    vertices = [np.arange(642), []]
    gamma = 1.
    n_features = len(M)
    epsilon = 10. / n_features
    K = np.exp(- M / epsilon)

    mt_vertices = mt.vertices[mt.vertices < 642]
    v1_vertices = v1.vertices[v1.vertices < 642]

    seed = 42
    n_samples_per_task = 50
    n_times = 20
    time0, time1 = 5, 15

    # Create the four categories of brain signals with different random seeds
    meg_v1_0 = generate_samples(n_samples_per_task, n_times, time0,
                                v1_vertices, M=M, seed=seed)
    meg_v1_1 = generate_samples(n_samples_per_task, n_times, time1,
                                v1_vertices, M=M, seed=seed + 1)
    meg_mt_0 = generate_samples(n_samples_per_task, n_times, time0,
                                mt_vertices, M=M, seed=seed + 2)
    meg_mt_1 = generate_samples(n_samples_per_task, n_times, time1,
                                mt_vertices, M=M, seed=seed + 3)

    # to avoid numerical errors with Sinkhorn, add 1e-3
    meg = np.concatenate((meg_v1_0, meg_v1_1, meg_mt_0, meg_mt_1)) + 1e-3

    # create labels for categories
    y_time = np.r_[2 * np.r_[n_samples_per_task * [0],
                             n_samples_per_task * [1]].tolist()]
    y_space = np.r_[2 * n_samples_per_task * [0], 2 * n_samples_per_task * [1]]

    betas = [0, 0.001, 0.01, 0.1, 0.5, 1., 2., 3., 5., 10.]
    experiment = dict(meg=meg, y_time=y_time, y_space=y_space, betas=betas)
    train_data = []
    n_samples, n_times, dimension = meg.shape
    params = dict(K=K, epsilon=epsilon, gamma=gamma, n_jobs=4,
                  n_gpu_devices=n_gpu_devices)
    precomputed = sta_matrix(meg, betas, **params)
    experiment["sta"] = dict()
    for beta, train_ in zip(betas, precomputed):
        train = train_.copy()
        # shift the distance to avoid negative values with large betas
        train -= train.min()
        tsne_data = TSNE(metric="precomputed").fit_transform(train)
        experiment["sta"][beta] = tsne_data

    method = "soft"
    experiment["soft"] = dict()
    for beta in betas:
        precomputed = sdtw_matrix(meg, beta, n_jobs=10)
        train = precomputed.copy()
        # shift the distance to avoid negative values with large betas
        train -= train.min()
        tsne_data = TSNE(metric="precomputed").fit_transform(train)
        experiment[method][beta] = tsne_data

    expe_file = open("data/tsne-brains.pkl", "wb")
    pickle.dump(experiment, expe_file)
