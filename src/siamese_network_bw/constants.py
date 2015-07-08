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

def get_output_cluster_path(graph_name):
    """
    Generates a cluster image path name, using the given graph name to build it up. This graph
    tests how well we are clustering faces.
    """
    return OUTPUT_LOG_PREFIX + "_" + graph_name + ".cluster.png"

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

SOLVER_FILE = ROOT_DIR + "/model/siamese_solver.prototxt"

TRAINED_MODEL = ROOT_DIR + "/model/siamese.prototxt"
TRAINED_WEIGHTS = ROOT_DIR + "/model/trained_model.caffemodel"

CAFFE_HOME = os.environ.get("CAFFE_HOME")

# Architecture string that will appear on graphs; good for relatively stable
# hyperparameter tuning.
ARCHITECTURE = "B&W; 20x boost; bigger filters + bigger hidden; min: 3, max: 10"
