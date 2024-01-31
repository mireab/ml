import numpy as np
from enum import StrEnum, auto

class ActivationFunctions(StrEnum):
    SIGMOID = auto()
    TANH = auto()
    ID = auto()

    def _call_(self, z):
        match self:
            case ActivationFunctions.SIGMOID:
                return (1 / (1 + np.exp(-z)))
            case ActivationFunctions.TANH:
                return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
            case ActivationFunctions.ID:
                return z


    def derivative(self, z):
        match self:
            case ActivationFunctions.SIGMOID:
                s = self(z)
                return s * (1 - s)
            case ActivationFunctions.TANH:
                return 1 - self(z)**2
            case ActivationFunctions.ID:
                return 1

    @property
    def threshold_value(self):
        match self:
            case ActivationFunctions.SIGMOID:
                return 0.5
            case ActivationFunctions.TANH:
                return 0