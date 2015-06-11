import os

def determine_output_ending():
    """
    We add a unique ending to our output files so they can stack over time,
    such as output0001.log. This method determines an appropriate, unused
    ending, incrementing through those that are already present.
    """

    file_found = False
    idx = 1
    while not file_found:
        if not os.path.isfile(LOG_DIR + "/output%04d.png" % (idx)):
          return "%04d" % (idx)
        idx += 1

WIDTH = 47
HEIGHT = 62

ROOT_DIR = "./src/siamese_network_bw"
LOG_DIR = ROOT_DIR + "/logs"

TRAINING_FILE = ROOT_DIR + "/data/siamese_network_train_leveldb"
VALIDATION_FILE = ROOT_DIR + "/data/siamese_network_validation_leveldb"
TESTING_FILE = ROOT_DIR + "/data/siamese_network_test_leveldb"

# The number to append to output files, such as output0001.log.
OUTPUT_ENDING = determine_output_ending()

# Log output file path.
OUTPUT_LOG_PATH = LOG_DIR + "/output" + OUTPUT_ENDING + ".log"

# Graph output file path.
OUTPUT_GRAPH_PATH = LOG_DIR + "/output" + OUTPUT_ENDING + ".png"

SOLVER_FILE = ROOT_DIR + "/model/siamese_solver.prototxt"

TRAINING_SIZE = 6560
VALIDATION_SIZE = 1640
TESTING_SIZE = 1000

CAFFE_HOME = os.environ.get("CAFFE_HOME")

# Architecture string that will appear on graphs; good for relatively stable
# hyperparameter tuning.
ARCHITECTURE = "B&W MNIST"
