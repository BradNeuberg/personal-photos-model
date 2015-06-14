import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np
import caffe

import constants as constants
import prepare_data as prepare_data
import siamese_network_bw.siamese_utils as siamese_utils

def predict(img_1, img_2):
  print "Predicting..."
  # TODO!!! Implement!!!

def test_cluster(weight_file=constants.TRAINED_WEIGHTS):
  """
  Tests a few people to see how they cluster, producing an image file.
  """
  print "Generating test cluster..."
  (data, labels, identities) = prepare_data.prepare_testing_cluster_data()

  print "\tInitializing Caffe using weight file %s..." % (weight_file)
  caffe.set_mode_cpu()
  net = caffe.Net(constants.TRAINED_MODEL, weight_file, caffe.TEST)

  print "\tRunning through network to generate cluster prediction..."
  out = net.forward_all(data=data)

  print "\tGraphing..."
  feat = out['feat']
  f = plt.figure(figsize=(16,9))
  c = {
    str(identities[0]): "#ff0000",
    str(identities[1]): "#ffff00",
    str(identities[2]): "#00ff00",
    str(identities[3]): "#00ffff",
    str(identities[4]): "#0000ff",
  }
  for i in range(len(feat)):
    plt.plot(feat[i, 0], feat[i, 1], ".", c=c[str(labels[i])])

  plt.grid()

  plt.savefig(constants.OUTPUT_CLUSTER_PATH)
  print("\t\tGraph saved to %s" % constants.OUTPUT_CLUSTER_PATH)
