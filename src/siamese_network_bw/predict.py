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

def test_clusters(data=None, weight_file=constants.TRAINED_WEIGHTS):
    """
    Tests a few people to see how they cluster across the training and validation data, producing
    image files.
    """
    print "Generating cluster details..."
    cluster_details = data.get_clustered_faces()

    print "\tInitializing Caffe using weight file %s..." % (weight_file)
    caffe.set_mode_cpu()
    net = caffe.Net(constants.TRAINED_MODEL, weight_file, caffe.TEST)

    test_cluster(net, cluster_details["train"], "train")
    test_cluster(net, cluster_details["validation"], "validation")

def test_cluster(net, cluster_details, graph_name):
    """
    Actually tests the cluster on the network for a given data set, generating a graph.
    """
    data = cluster_details["data"]
    target = cluster_details["target"]
    good_identities = cluster_details["good_identities"]

    print "\tRunning through network to generate cluster prediction for %s dataset..." % graph_name
    out = net.forward_all(data=data)

    print "\tGraphing..."
    feat = out["feat"]
    f = plt.figure(figsize=(16,9))
    c = {
      str(good_identities[0]): "#ff0000",
      str(good_identities[1]): "#ffff00",
      str(good_identities[2]): "#00ff00",
      str(good_identities[3]): "#00ffff",
      str(good_identities[4]): "#0000ff",
    }
    for i in range(len(feat)):
      plt.plot(feat[i, 0], feat[i, 1], ".", c=c[str(target[i])])
    plt.grid()

    plt.savefig(constants.get_output_cluster_path(graph_name))
    print("\t\tGraph saved to %s" % constants.get_output_cluster_path(graph_name))

def test_validation_pairings(data=None, weight_file=constants.TRAINED_WEIGHTS):
    """
    Computes the Euclidean distance for all pairings of the validation data against the trained
    contrastive loss function. Compares this against a threshold to generate an ROC curve and
    confusion matrix.
    """
    print "Testing validation pairings..."
    validation = data.get_validation_images()
    data = validation["data"]
    target = validation["target"]

    print "\tInitializing Caffe using weight file %s..." % (weight_file)
    caffe.set_mode_cpu()
    net = caffe.Net(constants.TRAINED_MODEL, weight_file, caffe.TEST)

    print "\tRunning through network to generate validation pairings..."
    out = net.forward_all(data=data)["feat"]

    # TODO: Break this into its own method.
    threshold = 80
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    actual_positive = 0
    actual_negative = 0
    for image_1_idx in range(len(out)):
      for image_2_idx in range(len(out)):
        actual_same = target[image_1_idx] == target[image_2_idx]
        if actual_same:
            actual_positive = actual_positive + 1
        else:
            actual_negative = actual_negative + 1

        distance = np.linalg.norm(out[image_1_idx] - out[image_2_idx])

        predicted_same = False
        if distance <= threshold:
            predicted_same = True

        if predicted_same == True and actual_same == True:
            true_positive = true_positive + 1
        elif predicted_same == False and actual_same == False:
            true_negative = true_negative + 1
        elif predicted_same == True and actual_same == False:
            false_positive = false_positive + 1
        elif predicted_same == False and actual_same == True:
            false_negative = false_negative + 1

    # TODO: Is there a Python library we can use to print a table?
    print "\n"
    print "\t\t\t\tPositive\t\tNegative"
    print "Positive (%d)\t\t\tTrue Positive (%d)\tFalse Positive (%d)" % \
        (actual_positive, true_positive, false_positive)
    print "Negative (%d)\t\tFalse Negative (%d)\tTrue Negative (%d)" % \
        (actual_negative, false_negative, true_negative)
    print "\n"

    actual_positive = float(actual_positive)
    actual_negative = float(actual_negative)
    true_positive = float(true_positive)
    true_negative = float(true_negative)
    false_positive = float(false_positive)
    false_negative = float(false_negative)

    # TODO: Break this into methods
    accuracy = (true_positive + true_negative) / (actual_positive + actual_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2.0 * ((precision * recall) / (precision + recall))

    print "Accuracy: %f" % accuracy
    print "Precision: %f" % precision
    print "Recall: %f" % recall
    print "F1 Score: %f" % f1_score

