Spatio-temporal Alignement
--------------------------

Guide to reproduce the results of the paper
"Spatio-temporal alignements: Optimal transort through space and time"
(https://arxiv.org/abs/1910.03860).



If your platform contains GPUs, please set the number of devices you would
like to use in the beginning the
scripts `run_tsne...`.

1. Installation
---------------

The implementation of the proposed STA is available in the package `sta`
provided in this folder. Before installing it, please make sure you have a
miniconda environment installed and the following
necessary dependencies (available through pip or conda):

- numpy
- cython
- joblib
- matplotlib
- scikit-learn
- soft-dtw (https://github.com/mblondel/soft-dtw/tree/master/sdtw)
- torch
- pandas
- numba
- pyts (https://pyts.readthedocs.io/en/latest/)

To reproduce the brain imaging experiment, you will also seed the MNE package (available with pip):

- mne (https://mne.tools/stable/index.html)

If you want 3D visualization of the brain signals, you also need

- mayavi
- pysurfer


Then proceed to the sta folder and run:

    python setup.py develop


2. Experiments
--------------

2.1 theoretical_bound
---------------------

* run `plot_example.py` to produce Figure 2 of the paper.
* run `plot_bound.py` to produce the theoretical bound (Figure 3)


2.2 Brain imaging
-----------------

0. Make sur `mne` is installed.
1. Open `run_tsne_brains.py` to set the `n_gpus` and `n_jobs` params and run it to reproduce Figure 5.
2. To reproduce Figure 4, verify your installation of mayavi and pysurfer and run `plot_brains.py`.
    This last step can eventually take time because the MNE-Sample data must be downloaded.

2.3 Handwrittern letters
------------------------

0. Run `process_chars.py` to generate and save the processed data.
1. Run `plot_chars.py` to visualize the chars (figure 6)
2. Open `run_tsne_brains.py` to set the `n_gpus` and `n_jobs` params and  to compute and save the tsne maps
3. run `plot_tsne_chars.py` to reproduce Figure 7.
