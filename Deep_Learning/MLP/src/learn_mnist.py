import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_net import Network
from utils import load_data_train_validation, load_data_test, generate_batch, train_epoch, evaluate, \
    plot_training_curves, show_predictions
from sgd import Sgd
import copy


if __name__ == "__main__":
    
    # Loading the data
    train_data, train_labels, val_data, val_labels = load_data_train_validation()
    test_data, test_labels = load_data_test()

    # Initializing the network and the optimizer
    model = Network()
    optimizer = Sgd(learning_rate=0.5, reg=0.00001)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_acc = 0.0
    best_model = None
    epochs = 10
    best_accuracy = 0

    for epoch in range(epochs):
        # Train
        batch_training_data, batched_training_labels = generate_batch(train_data, train_labels, batch_size=32, shuffle=True)
        train_loss, train_acc = train_epoch(batch_training_data, batched_training_labels, model=model, optimizer=optimizer)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        print(f"Epoch {epoch}, Training, loss = {train_loss}, accuracy = {train_acc}")

        # Evaluate
        batched_val_data, batched_val_labels = generate_batch(val_data, val_labels, batch_size=32, shuffle=True)
        val_loss, val_acc = evaluate(batched_val_data, batched_val_labels, model)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        print(f"Epoch {epoch}, Validation, loss = {val_loss}, accuracy = {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)

    batch_test_data, batch_test_labels = generate_batch(test_data, test_labels, batch_size=32)
    _, test_acc = evaluate(batch_test_data, batch_test_labels, model)

    print(f"Accuracy on test data: {test_acc:.4f}")
    plot_training_curves(train_loss_history, train_acc_history, val_loss_history, val_acc_history)

    sample_input = batch_test_data[0][0:5]
    _, _, pred = model.forward_backward(sample_input, mode="eval")
    show_predictions(sample_input, pred)

