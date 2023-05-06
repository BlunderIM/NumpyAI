
import numpy as np

class Optimizer:
    def __init__(self, learning_rate: float = 1e-4, reg: float =1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def apply_regularization(self, model):
        """
        Apply L2 regularization penalty

        Args:
            model (_MlpNetwork): The model with the gradients
        Returns:
            None
        """
        # Objective = CE loss + reg/2 * sum over (weights^2)
        # Gradient becomes -> reg + learning_rate * weight

        for key, value in model.gradients.items():
            if 'W' in key:
                model.gradients[key] += self.reg * model.params[key]