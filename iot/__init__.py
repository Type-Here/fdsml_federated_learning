"""Inference-time half of the project: the deployed model under input shift.

Nothing here takes part in federated training. This package starts from a
finished global checkpoint. It studies what happens when the images it sees at
test time are corrupted (weather, sensor noise, blur), and how cheaply a device
can adapt to that without ever computing a gradient.

Layout, and the reason for it:

    corruption_shim.py   `imagecorruptions` on a modern numpy / scikit-image
    gtsrb_c.py           builds the corrupted test set on disk
    routing.py           which normalization state to load, or refuse to
    metrics.py           what the counts mean, intervals included
    bn_bank.py           extracts the states and descriptors from the network
    bn_adapt.py          the baselines: blind adaptation, or a loaded state
    source_model.py      the checkpoint rebuilt, and its BatchNorm recalibrated
    stream_eval.py       the experiment: four arms over the corrupted stream

Modules that do not need `torch` are kept separate from the ones that do, so the
maths can be unit-tested on a machine with no GPU and no torch installed.
"""
