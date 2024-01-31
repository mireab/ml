import numpy as np
from nn.InitStrategies import InitStrategies
from nn.ActivationFunctions import ActivationFunctions


class Layer:
    def __init__(self, unit_number):
        if not isinstance(unit_number, int) or unit_number <= 0:
            raise ValueError("unit_number must be a positive integer")
        self.unit_number = unit_number
        self.random_seed = None
        self._output = None
        self._z = None
        self._previous_weights_update = 0
        
    @property
    def _input_size(self):
        return self._previous_layer.unit_number
    
    @property
    def _input(self):
        return self._previous_layer._output
    
    
    def _initialize_weights(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.weights = self.init_strategy.init_weights(self._input_size, self.unit_number)
        self.biases = self.init_strategy.init_biases(self._input_size, self.unit_number)
    

class InputLayer(Layer):
    def __init__(self, unit_number):
        if not isinstance(unit_number, int) or unit_number <= 0:
            raise ValueError("Unit_number must be a positive integer, but received {}".format(unit_number))
        super().__init__(unit_number)
        self._next_layer = None
        self._value = None
        self._output = None
    
    def _set_input(self, input):
        assert len(input) == self.unit_number, "Input shape: {} doesn't match the first layer's input size: {}".format(input.shape, self.unit_number)
        self._value=input

    def _forward(self):
        self._output=self._value


class HiddenLayer(Layer):
    def __init__(self, unit_number, activation_function = ActivationFunctions.SIGMOID, init_strategy = InitStrategies.RANDOM):
        super().__init__(unit_number)
        self.weights = None
        self.biases = None
        self.activation_function = activation_function
        self.init_strategy = init_strategy
        self._delta = None

    def _compute_gradient(self, *_):
        self._delta = self.activation_function.derivative(self._z) * np.dot(self._next_layer.weights, self._next_layer._delta)
        weights_gradient = np.outer(self._input, self._delta)
        biases_gradient = self._delta
        if weights_gradient.shape != self.weights.shape:
            raise ValueError(f"Gradient shape: {weights_gradient.shape} does not match self.weights shape: {self.weights.shape}.")
        return weights_gradient, biases_gradient


    def _forward(self):
        self._z = np.dot(self._input, self.weights) + self.biases
        self._output = self.activation_function(self._z)


class OutputLayer(Layer):
    def __init__(self, unit_number, activation_function = ActivationFunctions.SIGMOID, init_strategy = InitStrategies.RANDOM):
        super().__init__(unit_number)
        self.weights = None
        self.biases = None
        self.activation_function = activation_function
        self.threshold_value = activation_function.threshold_value
        self.init_strategy = init_strategy
        self._delta = None
        self._sq_error = []

    # AGGIUNTO SELF._error
    def _compute_gradient(self, pattern_label):
        self._error = (pattern_label - self._output)
        self._sq_error.append(self._error**2)
        self._delta = self._error * self.activation_function.derivative(self._z)
        weights_gradient = np.outer(self._input, self._delta)
        biases_gradient = self._delta
        if weights_gradient.shape != self.weights.shape:
            raise ValueError("Gradient shape: {} does not match self.weights shape: {}.".format(weights_gradient.shape, self.weights.shape))
        return weights_gradient, biases_gradient

    def _forward(self):
        self._z = np.dot(self._input, self.weights) + self.biases
        self._output = self.activation_function(self._z)
