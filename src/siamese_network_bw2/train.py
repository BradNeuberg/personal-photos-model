import shutil
import subprocess
import sys
import csv

import constants as constants
import graph as graph

def train(output_graphs):
    print("Training data, generating graphs: %r" % output_graphs)

    run_trainer()
    generate_parsed_logs()
    (training_details, validation_details) = parse_logs()
    graph.plot_results(training_details, validation_details)

def run_trainer():
    """
    Runs Caffe to train the model.
    """

    print("\tRunning trainer...")
    with open(constants.OUTPUT_LOG_PATH, "w") as f:
        process = subprocess.Popen([constants.CAFFE_HOME + "/build/tools/caffe", "train",
            "--solver=" + constants.SOLVER_FILE],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            f.write(line)

        print("\t\tTraining output saved to %s" % constants.OUTPUT_LOG_PATH)

def generate_parsed_logs():
    """
    Takes the raw Caffe output created while training the model in order
    to generate reduced statistics, such as giving iterations vs. test loss.
    """

    print("\tParsing logs...")
    process = subprocess.Popen([constants.CAFFE_HOME + "/tools/extra/parse_log.sh",
        constants.OUTPUT_LOG_PATH], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)

    # The parse_log.sh script dumps its output in our root; move it to a better location.
    shutil.rmtree(constants.OUTPUT_LOG_PATH + ".train", ignore_errors=True)
    shutil.rmtree(constants.OUTPUT_LOG_PATH + ".validate", ignore_errors=True)
    shutil.move("output" + constants.OUTPUT_ENDING + ".log.train",
        constants.OUTPUT_LOG_PATH + ".train")
    shutil.move("output" + constants.OUTPUT_ENDING + ".log.test",
        constants.OUTPUT_LOG_PATH + ".validate")

    logs = [
        {"title": "Testing", "filename": "train"},
        {"title": "Validation", "filename": "validate"}
    ]
    for log in logs:
        print("\n\t\tParsed %s log:" % log["title"])
        with open(constants.OUTPUT_LOG_PATH + "." + log["filename"], "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                print("\t\t\t%s" % line)

    print("\t\tParsed training log saved to %s" % (constants.OUTPUT_LOG_PATH + ".train"))
    print("\t\tParsed validation log saved to %s\n" % (constants.OUTPUT_LOG_PATH + ".validate"))

def parse_logs():
    training_iters = []
    training_loss = []
    for line in csv.reader(open(constants.OUTPUT_LOG_PATH + ".train"), delimiter=" ",
                            skipinitialspace=True):
        # Skip first line, which has column headers.
        if line[0] == "#Iters":
            continue

        training_iters.append(int(line[0]))
        training_loss.append(float(line[2]))

    validation_iters = []
    validation_accuracy = []
    for line in csv.reader(open(constants.OUTPUT_LOG_PATH + ".validate"), delimiter=" ",
                            skipinitialspace=True):
        # Skip first line, which has column headers.
        if line[0] == "#Iters":
            continue

        validation_iters.append(int(line[0]))
        validation_accuracy.append(float(line[2]))

    return ({
        "iters": training_iters,
        "loss": training_loss
    }, {
        "iters": validation_iters,
        "accuracy": validation_accuracy
    })
