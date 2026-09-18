"""
experiments/dropout.py — A4: random detection dropout (sparse-landmark regime).

Deterministic and independent of every other random stream: each frame draws from its own
generator seeded by (seed, frame_id), so dropping detections on one frame cannot shift the random
state of any later frame or of the pipeline's own seeded libraries.
"""

import numpy as np


def apply_detection_dropout(centroids, classes, confs, bboxes, rate, seed, frame_id):
    """Drop each detection independently with probability `rate`. Order of survivors is preserved."""
    n = len(centroids)
    if n == 0 or rate <= 0.0:
        return centroids, classes, confs, bboxes
    keep = np.random.default_rng([int(seed), int(frame_id)]).random(n) >= float(rate)
    pick = [i for i in range(n) if keep[i]]
    return ([centroids[i] for i in pick], [classes[i] for i in pick],
            [confs[i] for i in pick], [bboxes[i] for i in pick])
