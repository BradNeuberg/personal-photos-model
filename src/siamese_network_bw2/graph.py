import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

import constants

def plot_results(training_details, validation_details):
    """
    Generates a combined training/validation graph. The graph has two y axis on either side:
    one for the training loss, and the other for the validation accuracy.
    """
    print "\tPlotting results..."
    fig, ax1 = plt.subplots()
    ax1.plot(training_details["iters"], training_details["loss"], "b-")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Training Loss", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    ax2.plot(validation_details["iters"], validation_details["accuracy"], "r-")
    ax2.set_ylabel("Validation Accuracy", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    legend_font = FontProperties()
    legend_font.set_size("small")
    blue_line = mpatches.Patch(color="blue", label="Training Loss")
    red_line = mpatches.Patch(color="red", label="Validation Accuracy")
    plt.legend(handles=[blue_line, red_line], prop=legend_font, loc="lower right")

    plt.suptitle("Iterations vs. Training Loss/Validation Accuracy", fontsize=14)
    plt.title(get_hyperparameter_details(), style="italic", fontsize=12)

    plt.savefig(constants.OUTPUT_GRAPH_PATH)
    print("\t\tGraph saved to %s" % constants.OUTPUT_GRAPH_PATH)

    plt.show()

def get_hyperparameter_details():
    """
    Parse out some of the values we need from the Caffe solver prototext file.
    """
    solver = open(constants.SOLVER_FILE, "r")
    details = solver.read()
    lr = re.search("^base_lr:\s*([0-9.]+)$", details, re.MULTILINE).group(1)
    max_iter = re.search("^max_iter:\s*([0-9.]+)$", details, re.MULTILINE).group(1)
    return "(lr: %s; max_iter: %s; arch: %s)" % (lr, max_iter, constants.ARCHITECTURE)
