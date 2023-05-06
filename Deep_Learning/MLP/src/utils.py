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
    data, label = load_csv("../data/mnist_train.csv")

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
    data, label = load_csv("../data/mnist_test.csv")

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
        data, labels = zip(*data_label_pairs)

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
        loss, accuracy, _ = model.forward_backward(input, target)
        optimizer.update(model)
        epoch_loss += loss
        correct_guesses += accuracy * input.shape[0]
        samples_count += input.shape[0]

    epoch_loss /= len(batched_training_data)
    epoch_accuracy = correct_guesses/samples_count
    
    return epoch_loss, epoch_accuracy


def evaluate(batched_data, batched_labels, model):
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
        loss, accuracy, _ = model.forward_backward(input, target, mode="eval")
        epoch_loss += loss
        correct_guesses += accuracy * input.shape[0]
        samples_count += input.shape[0]

    epoch_loss /= len(batched_data)
    epoch_accuracy = correct_guesses/samples_count
    
    return epoch_loss, epoch_accuracy

def show_predictions(sample_input, sample_labels):
    """
    Take sample input and model and then save an image showign the results

    Args:
        sample_input (list): list of numpy arrays representing the input images
        sample_labels (list): list of labels
    Returns:
        None
    """
    fig = plt.figure(constrained_layout=True, figsize=(10, 4));
    fig.set_facecolor('white')
    gs = GridSpec(2, 5, figure=fig);
    subplots_list = []
    for i in range(5):
        subplots_list.append(fig.add_subplot(gs[0, i]))
        subplots_list.append(fig.add_subplot(gs[1, i]))
    subplot_idx = 0
    for i in range(5):
        subplots_list[subplot_idx].imshow(sample_input[i].reshape(28, 28), cmap="gray")
        subplots_list[subplot_idx + 1].text(0.5, 0.4, str(sample_labels[i]), fontsize=96, color='royalblue',
                                            ha='center', va='center')
        subplot_idx += 2
    subplots_list[0].set_ylabel('Input')
    subplots_list[1].set_ylabel('Prediction')
    for p in subplots_list:
        p.set_xticks([])
        p.set_yticks([])

    fig.savefig('../data/demo.png')


def plot_training_curves(training_loss_history, training_acc_history, validation_loss_history, validation_acc_history):
    """
    Plot learning curves and saves the file

    Args:
        training_loss_history (list): Training loss for each epoch
        training_acc_history (list): Training accuracy for each epoch
        validation_loss_history (list): Validation loss for each epoch
        validation_acc_history (list): Validation loss for each epoch
    Returns:
        None
    """
    fig = plt.figure(constrained_layout=True, figsize=(12, 5));
    gs = GridSpec(1, 2, figure=fig);
    fig.set_facecolor('white');
    ax1 = fig.add_subplot(gs[0, 0]);
    ax2 = fig.add_subplot(gs[0, 1]);
    epochs = list(range(1, len(training_acc_history)+1))
    ax1.plot(epochs, training_acc_history, color='royalblue');
    ax1.plot(epochs, validation_acc_history, color='darkorange');
    ax1.scatter(epochs, training_acc_history, color='royalblue');
    ax1.scatter(epochs, validation_acc_history, color='darkorange');
    ax1.set(xlabel="Epochs");
    ax1.set(ylabel="Accuracy %");
    ax1.set(title="Accuracy Curve");
    ax1.grid();
    ax1.margins(x=0.05);
    ax1.legend(["Training", "Validation"]);

    ax2.plot(epochs, training_loss_history, color='royalblue');
    ax2.plot(epochs, validation_loss_history, color='darkorange');
    ax2.scatter(epochs, training_loss_history, color='royalblue');
    ax2.scatter(epochs, validation_loss_history, color='darkorange');
    ax2.set(xlabel="Epochs");
    ax2.set(ylabel="Loss");
    ax2.set(title="Loss Curve");
    ax2.grid();
    ax2.margins(x=0.05);
    ax2.legend(["Training", "Validation"]);

    fig.savefig('../data/Learning_Curves.png');





