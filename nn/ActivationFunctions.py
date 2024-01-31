import numpy as np
from enum import StrEnum, auto

class ActivationFunctions(StrEnum):
    SIGMOID = auto()
    TANH = auto()

    def __call__(self, z):
        match self:
            case ActivationFunctions.SIGMOID:
                return (1 / (1 + np.exp(-z)))
            case ActivationFunctions.TANH:
                return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))


    def derivative(self, z):
        match self:
            case ActivationFunctions.SIGMOID:
                s = self(z)
                return s * (1 - s)
            case ActivationFunctions.TANH:
                return 1 - self(z)**2

    @property
    def threshold_value(self):
        match self:
            case ActivationFunctions.SIGMOID:
                return 0.5
            case ActivationFunctions.TANH:
                return 0