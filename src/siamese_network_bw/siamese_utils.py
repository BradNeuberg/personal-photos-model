import numpy as np

def mean_normalize(entry):
  """
  Mean normalizes a pixel vector; entry is an unrolled pixel vector with
  two side by side facial images.
  """
  entry -= np.mean(entry, axis=0)
  entry *= (1.0/255.0)
  return entry

def normalize_target(target):
  """
  The siamese network contrastive loss function uses 0 for similar images, 1 for different.
  The LFW database is opposite. Flip this.
  """
  return abs(target - 1)

def get_key(idx):
  """
  Each image pair is a top level key with a keyname like 00059999, in increasing
  order starting from 00000000.
  """
  return "%08d" % (idx,)
