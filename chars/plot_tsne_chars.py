from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas as pd


fontsize = 13
params = {'axes.labelsize': fontsize + 2,
          'font.size': fontsize + 4,
          'legend.fontsize': fontsize + 1,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'pdf.fonttype': 42}
plt.rcParams.update(params)

expe_file = open("data/tsne-chars.pkl", "rb")
keys = np.array((pd.read_csv("data/keys.csv").columns))
experiment = pickle.load(expe_file)
y = experiment["y"].astype(int)
markers = ["$%s$" % s for s in np.unique(keys[y - 1])]
y = y - y.min()
methods = ["soft", "sta"]
titles = ["soft-DTW", "STA"]

betas = experiment["betas"]
betas = [0.1]  # Main figure of the paper
betas = [0, 0.1, 1.]  # supplementary figure

colors = ["indianred", "cornflowerblue", "gold", "purple", "navy",
          "forestgreen", "pink", "cyan", "black", "orange", "gray",
          "lightblue", "green", "brown", "red", "magenta"]
colors = np.array(colors)
f, axes = plt.subplots(len(betas), 2, figsize=(8, len(betas) * 4))
if len(betas) == 1:
    axes = [axes]
for i, (ax_row, beta) in enumerate(zip(axes, betas)):
    for j, (ax, method, title) in enumerate(zip(ax_row, methods, titles)):
        tsne_data = experiment[method][beta]
        for letter, marker, color in zip(np.unique(y), markers, colors):
            ids = np.where(y == letter)[0]
            ax.scatter(tsne_data[ids, 0], tsne_data[ids, 1], color=color,
                       alpha=0.5, marker=marker)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(title)
        if len(betas) > 1 and j == 0:
            ax.set_ylabel(r"$\beta = %s$" % (np.round(beta, 1)))
if len(betas) > 1:
    plt.savefig("fig/chars-tsne-all.pdf")
else:
    plt.savefig("fig/chars-tsne.pdf")
