import numpy as np
from _mlp_network import _MlpNetwork

class Network(_MlpNetwork):
    """
    Implement a two layer neural network
    """
    def __init__(self, input_size: float = 28*28, num_classes: int = 10, hidden_size: int = 128):
        super().__init__(input_size, num_classes)
        self.hidden_size = hidden_size
        self._param_initialization()

    def _param_initialization(self):
        """
        Initialize the weights of the network
        """
        self.params['b1'] = np.zeros(self.hidden_size)
        self.params['b2'] = np.zeros(self.num_classes)
        self.params['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        self.params['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # Initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward_backward(self, X, y=None, mode="train"):
        """
        Implements the forward pass. The forward pass returns the loss, accuracy and updates the gradient dict
        
        Args:
            X (numpy.ndarray): Batch of images
            y (numpy.ndarray): Labels of the images
            mode (string): If value is "train" then the model computes the gradients and adds them to the gradient dict
        Returns:
            loss (float): Loss of the batch
            accuracy (float): Accuracy of the batch
            predictions (numpy.ndarray): Tensor of predictions
        """
        loss = None
        accuracy = None
        # Forward pass
        weights_1, weights_2 = self.params['W1'], self.params['W2']
        bias_1, bias_2 = self.params['b1'], self.params['b2']
        mlp = _MlpNetwork()

        z1 = np.matmul(X, weights_1) + bias_1
        a1 = mlp.sigmoid(z1)
        z2 = np.matmul(a1, weights_2) + bias_2
        p = mlp.softmax(z2)
        predictions = np.argmax(p, axis=-1)

        loss = mlp.cross_entropy_loss(p, y)

        if y is not None:
            accuracy = mlp.compute_accuracy(p, y)

        # Backward pass
        if mode == "train":
            number_of_examples, number_of_classes = p.shape[0], p.shape[1]

            # Derivative of the loss wrt z_2 is the prediction minus the true probability
            y_tensor = np.zeros((number_of_examples, number_of_classes))
            y_tensor[np.arange(number_of_examples), y[:]] = 1

            # dL/dp * dp/dz2 = dl/dz2
            dz2 = p - y_tensor
            dz2 /= number_of_examples

            # dL/dW2 = dL/dz2 * dz2/dw2
            dW2 = np.matmul(a1.T, dz2)
            # dL/db2 = dL/dz2 * dz2/db2
            dz2_db2 = np.ones(dz2.shape)
            db2 = np.multiply(dz2, dz2_db2)
            # Squashing/Aggregating the bias across the batch
            db2 = np.sum(db2, axis=0)

            self.gradients['W2'] = dW2
            self.gradients['b2'] = db2


            # dL/da1 = dL/dz2 * dz2/da1 (dz2/da1 --> w2)
            da1 = np.matmul(dz2, weights_2.T)
            # dL/dz1 = dL/da1 * da1/dz1
            da1_dz1 = mlp.sigmoid_derivative(z1)
            dz1 = np.multiply(da1, da1_dz1)

            #dW1 = dL/dz1 * dz1/dW1
            dW1 = np.matmul(dz1.T, X).T
            # dL/db1 = dL/dz1 * dz1/db1
            dz1_db1 = np.ones(dz1.shape)
            db1 = np.multiply(dz1, dz1_db1)
            # Squashing the bias across the entire batch
            db1 = np.sum(db1, axis=0)

            self.gradients['W1'] = dW1
            self.gradients['b1'] = db1

        return loss, accuracy, predictions







