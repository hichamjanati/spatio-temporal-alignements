import numpy as np
from sklearn.manifold import TSNE
import pickle
import time
import torch
from sta import sta_matrix, sdtw_matrix
from sta.utils import cost_matrix, tonumpy


# change this if you have GPUs
# in our platform, this experiment ran on 4 GPUs in around 8 minutes

n_gpu_devices = 0

if __name__ == "__main__":

    t = time.time()
    X = np.load("data/chars-processed.npy")
    y = np.load("data/chars-labels-processed.npy")
    n_samples, n_times, dimension, _ = X.shape
    epsilon = 1 / dimension
    gamma = 1.

    # create ground metrics. M corresponds to the convolutional distance
    # matrix for convolutional sinkhorn

    _, M = cost_matrix(dimension)
    K = tonumpy(torch.exp(- M / epsilon))

    betas = [0, 0.001, 0.01, 0.1, 0.5, 1., 2., 3., 5., 10.]
    experiment = dict(X=X, y=y, betas=betas, epsilon=epsilon, gamma=gamma)
    train_data = []
    params = dict(K=K, epsilon=epsilon, gamma=gamma, n_jobs=4,
                  n_gpu_devices=n_gpu_devices)

    # compute sta distance matrix
    precomputed = sta_matrix(X, betas, **params)
    experiment["sta"] = dict()
    for beta, train_ in zip(betas, precomputed):
        train = train_.copy()
        # shift the distance to avoid negative values
        train -= train.min()
        tsne_data = TSNE(metric="precomputed").fit_transform(train)
        experiment["sta"][beta] = tsne_data

    # compute soft-dtw distance matrix
    method = "soft"
    experiment["soft"] = dict()
    for beta in betas:
        precomputed = sdtw_matrix(X, beta, n_jobs=10)
        train = precomputed.copy()
        # shift the distance to avoid negative values
        train -= train.min()
        tsne_data = TSNE(metric="precomputed").fit_transform(train)
        experiment[method][beta] = tsne_data

    # save all
    expe_file = open("data/tsne-chars.pkl", "wb")
    pickle.dump(experiment, expe_file)
    t = time.time() - t
    print("Full time: ", t)
