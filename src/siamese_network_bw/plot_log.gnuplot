#! /usr/bin/gnuplot

# These snippets serve only as basic examples.
# Customization is a must.
# You can copy, paste, edit them in whatever way you want.
# Be warned that the fields in the training log may change in the future.
# You had better check the data files before designing your own plots.

# Please generate the neccessary data files with
# /path/to/caffe/tools/extra/parse_log.sh before plotting.
# Example usage:
#     ./parse_log.sh mnist.log
# Now you have mnist.log.train and mnist.log.test.
#     gnuplot mnist.gnuplot

# The fields present in the data files that are usually proper to plot along
# the y axis are test accuracy, test loss, training loss, and learning rate.
# Those should plot along the x axis are training iterations and seconds.
# Possible combinations:
# 1. Test accuracy (test score 0) vs. training iterations / time;
# 2. Test loss (test score 1) time;
# 3. Training loss vs. training iterations / time;
# 4. Learning rate vs. training iterations / time;
# A rarer one: Training time vs. iterations.

# What is the difference between plotting against iterations and time?
# If the overhead in one iteration is too high, one algorithm might appear
# to be faster in terms of progress per iteration and slower when measured
# against time. And the reverse case is not entirely impossible. Thus, some
# papers chose to only publish the more favorable type. It is your freedom
# to decide what to plot.

reset
set terminal png
set output "`echo $GRAPHS_DIR/$FILENAME_PREFIX$FILENAME_SUFFIX.png`"
set style data lines
set key right

###### Fields in the data file your_log_name.log.train are
###### Iters Seconds TrainingLoss LearningRate

# Both training and validation loss & accuracy vs. training loss.
set title "Iterations vs. Loss/Accuracy `echo $DETAILS`"
set xlabel "Iterations"
set ylabel "Loss/Accuracy"
plot "`echo $LOG_DIR`/output.log.train" using 1:3 title "training (loss)", \
     "`echo $LOG_DIR`/output.log.test" using 1:3 title "validation (accuracy)"

# Training loss vs. training iterations
#set title "Training loss vs. training iterations"
#set xlabel "Training loss"
#set ylabel "Training iterations"
#plot "output.log.train" using 1:3 title "output"

# Training loss vs. training time
# plot "output.log.train" using 2:3 title "output"

# Learning rate vs. training iterations;
# plot "output.log.train" using 1:4 title "output"

# Learning rate vs. training time;
# plot "output.log.train" using 2:4 title "output"


###### Fields in the data file your_log_name.log.test are
###### Iters Seconds TestAccuracy TestLoss

# Test loss vs. training iterations
# plot "output.log.test" using 1:4 title "output"

# Test accuracy vs. training iterations
# plot "output.log.test" using 1:3 title "output"

# Test loss vs. training time
# plot "output.log.test" using 2:4 title "output"

# Test accuracy vs. training time
# plot "output.log.test" using 2:3 title "output"
