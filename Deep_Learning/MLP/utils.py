import time
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from neural_net import Network

def load_csv(path):
    """
    Loads csv of data file

    Args:
        path (string): path of the csv file
    Returns:
        data (list): image data
        labels (list): labels of the image data
    """
    data = []
    labels = []

    with open(path, 'r') as fp:
        images = fp.readlines()[1:]
        images = [img.rstrip() for img in images]

    for image in images:
        data_as_list = image.split(',')
        # first value is the label
        y = int(data_as_list[0])
        x = data_as_list[1:]
        x = [int(pixel)/255 for pixel in x]
        data.append(x)
        labels.append(y)
    
    return data, labels

def load_data_train_validation():
    """
    Load data and labels for training and validation

    Args:
        None
    Returns:
        train_data (list): List of training data
        train_labels (list): List of training labels
        val_data (list): List of validation data
        val_labels (list): List of validation labels
    """
    print("### Loading Training  Data ###")
    data, label = load_csv("./mnist/mnist_train.csv")

    data_size = len(data)
    # 80% of the data for training
    train_data = data[0:int(data_size*0.8)]
    train_labels = label[0:int(data_size*0.8)]
    # 20% of the data for validation
    validation_data = data[int(data_size*0.8):]
    validation_labels = label[int(data_size*0.8):]

    return train_data, train_labels, validation_data, validation_labels


def load_data_test():
    """
    Load data and labels for testing

    Args:
        None
    Returns:
        test_data (list): List of testing data
        test-_labels (list): List of testing labels
    """
    print("### Loading Testing Data ###")
    data, label = load_csv("./mnist/mnist_test.csv")

    return data, label

def generate_batch(data, labels, batch_size=32, shuffle=True):
    """
    Produce a batch given data and labels

    Args:
        data (list): List containing the data
        labels (list): List containing the labels
        batch_size (int): Batch size
        shuffle (bool): Conditional to shuffle data or not 
    Returns:
        batched_data (list): List of numpy arrays
        batched_labels (list): List of targets
    """
    if shuffle:
        data_label_pairs = list(zip(data, labels))
        random.shuffle(data_label_pairs)
        data, label = zip(*data_label_pairs)

    num_batches = np.ceil(len(data)/batch_size)
    batched_data, batched_labels = [], []
    batch_counter, counter = 0, 0
    while batch_counter < num_batches:
        batched_data.append(np.array(data[counter:counter+batch_size]))
        batched_labels.append(np.array(labels[counter:counter+batch_size]))
        counter += batch_size
        batch_counter += 1

    return batched_data, batched_labels


def train_epoch(batched_training_data, batched_training_labels, model, optimizer):
    """
    Train one episode by doing forward pass, computing the loss, doing a backward pass and then update the weights
    
    Args:
        batched_training_data (list): List containing numpy arrays
        batched_training_labels (list): List containing targets
        model (_MlpNetwork): The network being trained
        optimizer (_Optimizer): The optimizer of the network
    Returns:
        epoch_loss (float): Average loss of the current episode
        epoch_accuracy (float): Accuracy of the current episode
    """

    epoch_loss = 0
    correct_guesses = 0
    samples_count = 0

    for idx, (input, target) in enumerate(zip(batched_training_data, batched_training_labels)):
        loss, accuracy = model.forward(input, target)
        optimizer.update(model)
        epoch_loss += loss
        correct_guesses += accuracy * input.shape[0]
        samples_count += input.shape[0]

    epoch_loss /= len(batched_training_data)
    epoch_accuracy = correct_guesses/samples_count
    
    return epoch_loss, epoch_accuracy


def evaluate_epoch(batched_data, batched_labels, model):
    """
    Evaluate the model on non-training data

    Args:
        batched_data (list): List of numpy arrays
        batched_labels (list): List of targets
        model (_MlpNetwork): The network being tested
    """

    epoch_loss = 0
    correct_guesses = 0
    samples_count = 0

    for idx, (input, target) in enumerate(zip(batched_data, batched_labels)):
        loss, accuracy = model.forward(input, target)
        epoch_loss += loss
        correct_guesses += accuracy * input.shape[0]
        samples_count += input.shape[0]

    epoch_loss /= len(batched_data)
    epoch_accuracy = correct_guesses/samples_count
    
    return epoch_loss, epoch_accuracy

def plot_training_curves(training_loss, training_acc, validation_loss, validation_acc):
    """
    Plot learning curves and saves the file

    Args:
        training_loss (list): Training loss for each epoch
        training_acc (list): Training accuracy for each epoch
        validation_loss (list): Validation loss for each epoch
        validation_acc (list): Validation loss for each epoch
    Returns:
        None
    """





