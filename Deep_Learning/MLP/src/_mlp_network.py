
import numpy as np

class _MlpNetwork:
    def __init__(self, input_size: int = 28*28, num_classes: int = 10) -> None:
        self.input_size = input_size
        self.num_classes = num_classes
        self.params = {}
        self.gradients = {}

    # def _weight_initialization(self):
    #     pass

    # def forward(self):
    #     pass
        
    def sigmoid(self, input):
        """
        Compute the sigmoid activation

        Args:
            input (numpy.ndarray): A tensor coming from previous layer
        Returns:
                numpy.ndarray: A tensor after applying sigmoid activation
        """
        out = 1 / (1 + np.exp(-1 * input))

        return out
        
    def sigmoid_derivative(self, input):
        """
        Compute the analytical derivative of the sigmoid activation

        Args:
            input: (numpy.ndarray): Tensor
        Returns:
                numpy.ndarray: Tensor
        """
        # Derivative of a sigmoid function f(x) -> f(x)*(1-f(x))
        function = self.sigmoid(input)
        derivative = function * (1 -function)

        return derivative
        
    def relu(self, input):
        """
        Compute the ReLU activation

        Args:
            input: (numpy.ndarray): A tensor
        Returns:
                numpy.ndarray: Tensor
        """
        out = input * (input > 0)

        return out
        
    def relu_derivative(self, input):
        """
        Compute the analytical derivative of the ReLU activation

        Args:
            input (numpy.ndarray): Tensor
        Returns:
                numpy.ndarray: Tensor
        """
        out = 1 * (input > 0)

        return out
        
    def softmax(self, scores):
        """
        Compute Softmax scores
        
        Args:
            scores (numpy.ndarray): A numpy tensor containing raw scores
        Returns:
                numpy.ndarrray: Softmax probabilities
        """
        exp_scores = np.exp(scores)
        prob = exp_scores/(np.sum(exp_scores, axis=-1, keepdims=True))

        return prob
        
    def cross_entropy_loss(self, pred, correct):
        """
        Compute the Cross-Entropy loss

        Args:
            pred (numpy.ndarray): Softmax probabilities computed
            correct (numpy.ndarray): Correct labels
        Returns:
                float: Cross-Entropy Loss
        """
        # CE loss is: -1 * sum over number of classes of (true probability * log(predicted probability))
        # True probability for wrong classes is 0, therefore we can skip the summation
        loss_tensor = -1 * np.log(pred[np.arange(pred.shape[0]), correct])
        loss = np.mean(loss_tensor)

        return loss
        
    def compute_accuracy(self, pred, correct):
        """
        Compute the accuracy of the batch

        Args:
            pred (numpy.ndarray): Probabilities from the model
            correct (numpy.ndarray): Correct labels
        Returns:
                float: accuracy of the batch
        """
        number_of_predictions = pred.shape[0]
        predicted_scalars = np.argmax(pred, axis=1)
        comparison = correct - predicted_scalars
        wrong_prediction_count = np.count_nonzero(comparison)
        correct_prediction_count = number_of_predictions - wrong_prediction_count
        acc = correct_prediction_count/number_of_predictions

        return acc



