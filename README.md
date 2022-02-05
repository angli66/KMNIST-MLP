# Hiragana Recognition with MLP
This repository contains the code for a simple MLP trained and tested on the KMNIST dataset, which contains 10 classes of Hiragana (Japanese character).

There are two experiments that can be run. The first one is `check_gradients`, which check the correctness of backpropagation by comparing the gradient calculated by backpropagation and by numerical approximation. The second one is `train_mlp`, which train the MLP on the dataset and reports its accuracy and loss on the test set. The train set will be automatically divided into train subset and a validation set (with the size ratio of 4:1). All of the hyperparameters can be adjusted in `config.yaml`.

To run the code, first run `get_data.sh` to download the data set. After adjusting hyperparameters in `config.yaml`, run the experiment either with

    python3 main.py --check_gradients
or

    python3 main.py --train_mlp
.

`visualization.ipynb` contains the code for visualizing the results generated from `train_mlp`. It will create plots for train/validation accuracy, train/validation loss and report test accuracy/loss.