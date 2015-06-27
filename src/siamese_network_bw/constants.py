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

# The number to append to output files, such as output0001.log.
OUTPUT_ENDING = determine_output_ending()

OUTPUT_LOG_PREFIX = LOG_DIR + "/output" + OUTPUT_ENDING

# Log output file path.
OUTPUT_LOG_PATH = OUTPUT_LOG_PREFIX + ".log"

# Graph output file path.
OUTPUT_GRAPH_PATH = OUTPUT_LOG_PREFIX + ".png"

# Graph where we test clustering.
OUTPUT_CLUSTER_PATH = OUTPUT_LOG_PREFIX + ".cluster.png"

SOLVER_FILE = ROOT_DIR + "/model/siamese_solver.prototxt"

TRAINED_MODEL = ROOT_DIR + "/model/siamese.prototxt"
TRAINED_WEIGHTS = ROOT_DIR + "/model/trained_model.caffemodel"

TRAINING_SIZE = 105859
VALIDATION_SIZE = 2646

CAFFE_HOME = os.environ.get("CAFFE_HOME")

# Architecture string that will appear on graphs; good for relatively stable
# hyperparameter tuning.
ARCHITECTURE = "B&W MNIST; 10x boost custom pairs"
