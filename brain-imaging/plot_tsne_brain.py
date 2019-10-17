from matplotlib import pyplot as plt
import pickle
import numpy as np
from matplotlib.lines import Line2D


fontsize = 11
params = {'axes.labelsize': fontsize + 2,
          'font.size': fontsize,
          'legend.fontsize': fontsize + 1,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'pdf.fonttype': 42}
plt.rcParams.update(params)


expe_file = open("data/tsne-brains.pkl", "rb")
experiment = pickle.load(expe_file)
y_time = experiment["y_time"]
y_space = experiment["y_space"]

methods = ["soft", "sta"]
betas = experiment["betas"]
betas = [0, 0.1]  # Main figure of the paper
# betas = [0, 0.1, 2.]  # supplementary figure

color_base = np.array(["indianred", "cornflowerblue"])
colors = color_base[y_time]
marker_base = np.array(["+", "o"])
markers = marker_base[y_space]

titles = ["Soft-DTW", "STA"]
color_names = [r"At time $t_1$", r"At time $t_2$"]
marker_names = ["In zone V1", "In zone MT"]

legend_colors = [Line2D([0], [0], color="w", markerfacecolor=color,
                 marker="s", label=name, markersize=12,)
                 for color, name in zip(color_base, color_names)]
legend_marker = [Line2D([0], [0], color="w", markerfacecolor="k",
                 markeredgecolor="k", linewidth=2,
                 marker=marker, markersize=12, label=name)
                 for marker, name in zip(marker_base, marker_names)]
legend_colors.extend(legend_marker)
f, axes = plt.subplots(len(betas), 2, figsize=(6, 3 * len(betas)))
for j, (ax_row, beta) in enumerate(zip(axes, betas)):
    for i, (ax, method, title) in enumerate(zip(ax_row, methods, titles)):
        method_data = experiment[method]
        tsne_data = method_data[beta]
        for point, color, marker in zip(tsne_data, colors, markers):
            ax.scatter(point[0], point[1], color=color, s=60,
                       marker=marker, alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(r"$\beta = %s$" % (np.round(beta, 1)))
        if j == 0:
            ax.set_title(title)
ax.legend(handles=legend_colors, loc=2, ncol=2, bbox_to_anchor=[-0.9, 0.0])
if len(betas) > 2:
    plt.savefig("fig/tsne-meg-all.pdf", tight_layout=True)
else:
    plt.savefig("fig/tsne-meg.pdf", tight_layout=True)
