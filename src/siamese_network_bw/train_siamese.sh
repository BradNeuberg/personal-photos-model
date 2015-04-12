#!/usr/bin/env sh

# Example, providing the generated graph suffix followed by optional details
# to also add to the plotted graph:
# ./src/siamese_network_bw/train_siamese.sh "_lr_0_001" "(lr = 0.001)"

CAFFE=/usr/local/caffe
ROOT_DIR=src/siamese_network_bw
export LOG_DIR=$ROOT_DIR/logs
export GRAPHS_DIR=$ROOT_DIR/graphs
export FILENAME_PREFIX=siamese_network_bw
export FILENAME_SUFFIX=$1
export DETAILS=$2

$CAFFE/build/tools/caffe train --solver=$ROOT_DIR/model/siamese_solver.prototxt  2>&1 | tee "$LOG_DIR/output.log"
$CAFFE/tools/extra/parse_log.sh $LOG_DIR/output.log
mv output.log* $LOG_DIR
echo "---Training Details" && cat $LOG_DIR/output.log.train
echo
echo "---Validation Details" && cat $LOG_DIR/output.log.test
gnuplot $ROOT_DIR/plot_log.gnuplot
open -a Preview $GRAPHS_DIR/$FILENAME_PREFIX$FILENAME_SUFFIX.png
