import os
import mne
from mne import SourceEstimate as STC
from simulation_VEP import generate_samples
import numpy as np
from mayavi import mlab


spacing = "ico3"
n_times = 20

subjects_dir = mne.datasets.sample.data_path() + "/subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir


if __name__ == "__main__":
    hemi = "lh"
    # load brain regions
    mt = mne.read_label("data/lh.MT.label")
    v1 = mne.read_label("data/lh.V1.label")

    vertices = [np.arange(642), []]
    mt_vertices = mt.vertices[mt.vertices < 642]
    v1_vertices = v1.vertices[v1.vertices < 642]

    seed = 42
    n_samples_per_task = 1
    time_v1, time_mt = 5, 15
    meg_v1 = generate_samples(n_samples_per_task, time_v1, v1_vertices[7:8],
                              smoothing_time=2,
                              seed=seed)
    meg_mt = generate_samples(n_samples_per_task, time_mt, mt_vertices[1:2],
                              smoothing_time=2,
                              seed=seed + 3)
    stc_v1 = STC(data=meg_v1[0].T, vertices=vertices, tmin=0, tstep=1)
    stc_mt = STC(data=meg_mt[0].T, vertices=vertices, tmin=0, tstep=1)
    lims = [0., 0.5, 1.5]
    surfer_kwargs = dict(subject="fsaverage", hemi="lh", background="white",
                         foreground='black', cortex=("gray", -1, 6, True),
                         smoothing_steps=30, time_label=None,
                         clim=dict(kind="value", lims=lims))
    view_v1 = dict(azimuth=-54, elevation=90, distance=432,
                   focalpoint=np.array([0., 0., 0.]), roll=-91)
    view_mt = dict(azimuth=-135, elevation=90, distance=432,
                   focalpoint=np.array([0., 0., 0.]), roll=83)
    names = ["v1", "mt"]
    views = [view_v1, view_mt]
    stcs = [stc_v1, stc_mt]
    labels = [v1, mt]
    tss = [[3, 4, 5, 6, 7], [13, 14, 15, 16, 17]]

    for stc, view, name, label, ts in zip(stcs, views, names, labels, tss):
        for t in ts:
            brain = stc.plot(initial_time=t, **surfer_kwargs)
            brain.show_view(view)
            engine = mlab.get_engine()
            module_manager = engine.scenes[-1].children[-1].children[0]
            module_manager = module_manager.children[0]
            sc_lut_manager = module_manager.scalar_lut_manager
            sc_lut_manager.scalar_bar.number_of_labels = 3
            sc_lut_manager.scalar_bar.label_format = '%.1f'
            sc_lut_manager.scalar_bar.unconstrained_font_size = True
            sc_lut_manager.label_text_property.font_size = 50
            brain.add_label(label, borders=True, color="green")
            mlab.savefig("fig/%s-%s-2.png" % (name, t))
