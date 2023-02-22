if [ -d "../runs" ]
then
    echo "Starting tensorboard"
    tensorboard --logdir ../runs
else
    echo "Creating folder ../runs"
    tensorboard --logdir ../runs
if