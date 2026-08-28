"""Inference-time half of the project: the deployed model under input shift.

Nothing here takes part in federated training. This package starts from a
finished global checkpoint and studies what happens when the images it sees at
test time are corrupted (weather, sensor noise, blur), and how cheaply a device
can adapt to that without ever computing a gradient.

Layout, and the reason for it:

    corruption_shim.py   `imagecorruptions` on a modern numpy / scikit-image
    gtsrb_c.py           builds the corrupted test set on disk

Modules that do not need torch are kept separate from the ones that do, so the
maths can be unit-tested on a machine with no GPU and no torch installed - the
same split that `fipa.py` and `aggregation_policy.py` use on the training side.
"""
