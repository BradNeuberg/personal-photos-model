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
  (data, labels) = prepare_data.prepare_testing_cluster_data()

  print "\tInitializing Caffe using weight file %s..." % (weight_file)
  caffe.set_mode_cpu()
  net = caffe.Net(constants.TRAINED_MODEL, weight_file, caffe.TEST)

  print "\tRunning through network to generate cluster prediction..."
  out = net.forward_all(data=data)

  print "\tGraphing..."
  feat = out['feat']
  f = plt.figure(figsize=(16,9))
  c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
       '#ff00ff', '#990000', '#999900', '#009900', '#009999']
  for i in range(10):
      plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
  plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
  plt.grid()
  plt.savefig(constants.OUTPUT_CLUSTER_PATH)
  print("\t\tGraph saved to %s" % constants.OUTPUT_CLUSTER_PATH)
