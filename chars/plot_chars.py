import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import pandas as pd


fontsize = 30
params = {'axes.labelsize': fontsize + 2,
          'font.size': fontsize,
          'legend.fontsize': fontsize + 1,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'pdf.fonttype': 42}
plt.rcParams.update(params)


def imshow(x, cmap=cm.Greys, axes=None, titles=None, letter=""):
    n_times, dimension, dimension = x.shape
    if axes is None:
        f, axes = plt.subplots(1, n_times, figsize=(2 * n_times, 2),
                               sharex=True, sharey=True)

    for j, ax in enumerate(axes.ravel()):
        img = x[j].T
        ax.imshow(img, origin="lower", cmap=cmap)
        if titles:
            ax.set_title(titles[j])
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(letter)
    return axes

# set to True to generate figure 6 of the paper
# set to False to generate the supplementary figure
plot_one_letter = False


if __name__ == "__main__":

    X = np.load("data/chars-processed.npy")
    y = np.load("data/chars-labels-processed.npy")
    n_samples, n_times, dimension, _ = X.shape
    keys = np.array((pd.read_csv("data/keys.csv").columns))
    n = 20
    if plot_one_letter:
        letter = "g"
        class_id = keys.tolist().index(letter) - 1
        cmap = "Reds"
        f, axes = plt.subplots(2, 5, figsize=(10, 4))
        samples = X[[class_id * n, class_id * n + 5], 1::2].copy()
        samples[samples < 0.1] = np.nan
        for i, (ax_row, img) in enumerate(zip(axes, samples)):
            if i == 0:
                titles = ["t = %s" % e for e in [0, 2, 4, 6, 8]]
            else:
                titles = None
            imshow(img, cmap=cmap, axes=ax_row, titles=titles)
        plt.savefig("fig/letter-g.pdf")
        plt.close("all")
    else:
        letters = ["a", "b", "c", "d", "e", "g", "h"]
        n_letters = len(letters)
        ids = [keys.tolist().index(letter) for letter in letters]
        f, axes = plt.subplots(n_letters, n_times,
                               figsize=(n_times * 3, n_letters * 3))
        for ax_row, letter, class_id in zip(axes, letters, ids):
            xx = X[class_id * n: class_id * n + 1].copy()
            for img in xx:
                img[img < 0.05] = np.nan
                if class_id == ids[0]:
                    titles = ["t = %s" % e for e in np.arange(n_times)]
                else:
                    titles = None
                imshow(img, cmap="Purples", axes=ax_row, titles=titles,
                       letter=letter)
        plt.savefig("fig/all-letters.pdf", tight_layout=True)
