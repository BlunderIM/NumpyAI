import numpy as np
from optimizer import Optimizer

class Sgd(Optimizer):
    def __init__(self, learning_rate: float = 1e-4, reg: float = 1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        """
        Update the parameters based on the gradients

        Args:
            model (_MlpNetwork): The model to be updated
        Returns:
            None
        """
        for key, value in model.params.items():
            model.params[key] = model.params[key] - self.learning_rate * model.gradients[key]
            