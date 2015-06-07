import numpy as np

import constants

def mean_normalize(entry):
    """
    Mean normalizes a pixel vector. Entry is an unrolled pixel vector with two side by side facial
    images.
    """
    entry -= np.mean(entry, axis=0)
    return entry

def get_key(idx):
    """
    Each image pair is a top level key with a keyname like 00059999, in increasing
    order starting from 00000000.
    """
    return "%08d" % (idx,)
