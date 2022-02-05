################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
import os
import pickle

import numpy as np
import yaml


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    return np.eye(num_classes)[labels]


def one_hot_decoding(y):
    return np.argmax(y, axis = 1)


def write_to_file(path, data):
    """
    Dumps pickled data into the specified relative path.

    Args:
        path: relative path to store to
        data: data to pickle and store
    """
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(train=True):
    """
    Load the data from disk

    Args:
        train: Load training data if true, else load test data

    Returns:
        Tuple:
            Images
            Labels
    """
    directory = 'train' if train else 'test'
    patterns = np.load(os.path.join('./data/', directory, 'images.npz'))['arr_0']
    labels = np.load(os.path.join('./data/', directory, 'labels.npz'))['arr_0']
    return patterns.reshape(len(patterns), -1), labels


def load_config(path):
    """
    Load the configuration from config.yaml

    Args:
        path: A relative path to the config.yaml file

    Returns:
        A dict object containing the parameters specified in the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def shuffle(dataset):
    X, y = dataset
    order = np.random.permutation(len(X))
    return X[order], y[order]


def generate_minibatches(dataset, batch_size=128):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def z_score_normalize(X, u=None, sd=None):
    """
    Performs z-score normalization on X.
    f(x) = (x - μ) / σ
        where
            μ = mean of x
            σ = standard deviation of x

    Args:
        X: the data to min-max normalize
        u: the mean to normalize X with
        sd: the standard deviation to normalize X with

    Returns:
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.

    """
    if u is None:
        u = np.mean(X, axis=0)
    if sd is None:
        sd = np.std(X, axis=0)
    return ((X - u) / sd), (u, sd)
