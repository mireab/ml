import numpy as np
from enum import StrEnum, auto

_DEFAULT_SCALE = 0.3


class InitStrategies(StrEnum):    
    RANDOM = auto()
    GLOROT = auto()

    def scale(self, input_size=None, unit_number=None):
        match self:
            case InitStrategies.RANDOM:
                return _DEFAULT_SCALE
            case InitStrategies.GLOROT:
                return np.sqrt(6 / (input_size + unit_number))

    def init_weights(self, input_size, unit_number):
        scale = self.scale(input_size, unit_number)
        return np.random.uniform(-scale, scale, size=(input_size, unit_number))

    def init_biases(self, input_size, unit_number):
        match self:
            case InitStrategies.RANDOM:
                return np.random.uniform(-_DEFAULT_SCALE, _DEFAULT_SCALE, unit_number)
            case InitStrategies.GLOROT:
                return np.zeros(unit_number)               