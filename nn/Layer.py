import numpy as np
from nn.InitStrategies import InitStrategies
from nn.ActivationFunctions import ActivationFunctions


class Layer:

    """
    Class representing a Layer in a Neural Network.

    Attributes:
        unit_number: (int) The number of units 

        random seed : (int) | default = None | If set, alllows reproducibility of experiments setting
                        a random seed to control stochasticity during operations as weights initialization

        _output : (np.array) | default = None |  The output vector of the layer resulting, for each
                    unit, from the application of activation funtion to the weighted sums: it has n_units len

        _z : (np.array) | default = None | Array containing weighted sum of each unit: it has unit_number len

        _previous_weights_update : (np.array) | default = 0 | Array of size equal to relative weights matrix size, 
                                    containing difference between new weights matrix and old weigths
                                    matrix at iteration i-1
                
        _previous_bias_update : (np.array) | default = 0 | Vector with len(n_units) elements, 
                                    containing difference between new biases vector and old biases
                                    vector at iteration i-1

        _previous_w_gradient : (np.array) | default = None | Matrix of size equal to weights matrix containing the
                                            computed weights gradient

        _previous_b_gradien : (np.array) | default = None | Vector of size equal to bias vector containing the
                                            computed bias gradient
        
        The `Layer` class serves as an abstract class, and its subclasses are used to create layer objects within a neural network.
    """

    def __init__(self, unit_number):
        if not isinstance(unit_number, int) or unit_number <= 0:
            raise ValueError("unit_number must be a positive integer")
        self.unit_number = unit_number
        self.random_seed = None
        self._output = None
        self._z = None
        self._previous_weights_update = 0
        self._previous_bias_update = 0
        self._previous_w_gradient = None
        self._previous_b_gradien = None
        
        
    @property
    def _input_size(self):
        return self._previous_layer.unit_number
    

    @property
    def _input(self):
        return self._previous_layer._output
    
    
    def _initialize_weights(self):
        
        """
        Function to initialize weights and biases of a Layer as self.weights and self.biases

        :param random seed: (int) | default = None | If set, alllows reproducibility of weights initialization
        """

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.weights = self.init_strategy.init_weights(self._input_size, self.unit_number)
        self.biases = self.init_strategy.init_biases(self._input_size, self.unit_number)
    

class InputLayer(Layer):

    """
    Subclass representing Input Layer

    Attributes:
        _next_layer: (Layer) | default = None 
    """


    def __init__(self, unit_number):
        if not isinstance(unit_number, int) or unit_number <= 0:
            raise ValueError("Unit_number must be a positive integer, but received {}".format(unit_number))
        super().__init__(unit_number)
        self._next_layer = None
        self._value = None
        self._output = None
    

    def _set_input(self, input):

        """
        Set the input values for the Input Layer
        
        :param input: (np.array) An input array with a shape matching the number of units in the layer
        """

        assert len(input) == self.unit_number, "Input shape: {} doesn't match the first layer's input size: {}".format(input.shape, self.unit_number)
        self._value=input


    def _forward(self):
         
        """
        Forward propagation through the InputLayer.

        This method sets the output of the InputLayer to be the same as its input values.
        """
         
        self._output=self._value


class HiddenLayer(Layer):

    """
    Subclass representing Hidden Layer

    Attributes:

        weights: (np.array) | default = None | Weights matrix connetting this layer to the previous one
                with size (input_size, unit_number)
        
        biases: (np.array) | default = None | Biases vector for the current layer with unit_number elements
        
        activation_funtion: (ActivationFunctions) | default = ActivationFunctions.SIGMOID 

        init_strategy: (InitStrategies) | default = InitStrategies.RANDOM | The weight initialization strategy to use

        _delta: (np.array) | default = None | The vector of the deltas of the layer, where delta has to be intended as
                the derivative of the pattern error w.r.t. the the layer _z (weighted sums vector)
    """
        

    def __init__(self, unit_number, activation_function = ActivationFunctions.SIGMOID, init_strategy = InitStrategies.RANDOM):
        super().__init__(unit_number)
        self.weights = None
        self.biases = None
        self.activation_function = activation_function
        self.init_strategy = init_strategy
        self._delta = None


    def _compute_gradient(self, *_):
        """
        Compute the gradients of weights and biases during backpropagation.

        :param _: Not used parameter in order to make the function flexible, used both by Output anf Hidden Layer.

        :return: (tuple) | A tuple containing weights_gradient matrix and biases_gradient vector.
        """
        self._delta = self.activation_function.derivative(self._z) * np.dot(self._next_layer.weights, self._next_layer._delta)
        weights_gradient = np.outer(self._input, self._delta)
        biases_gradient = self._delta
        if weights_gradient.shape != self.weights.shape:
            raise ValueError(f"Gradient shape: {weights_gradient.shape} does not match self.weights shape: {self.weights.shape}.")
        return weights_gradient, biases_gradient


    def _forward(self):
        """
        Perform forward propagation through the HiddenLayer.

        This method computes the weighted sum of inputs, applies the activation function, and sets the output.
        """
        self._z = np.dot(self._input, self.weights) + self.biases
        self._output = self.activation_function(self._z)


class OutputLayer(Layer):
    
    """
    Subclass representing Output Layer

    Attributes (those seen above are excluded)
        _sq_error: (list of size(number of input patterns)) | default = [] | List to store squared errors
    """


    def __init__(self, unit_number, activation_function = ActivationFunctions.SIGMOID, init_strategy = InitStrategies.RANDOM):
        super().__init__(unit_number)
        self.weights = None
        self.biases = None
        self.activation_function = activation_function
        self.threshold_value = activation_function.threshold_value
        self.init_strategy = init_strategy
        self._delta = None
        self._sq_error = []


    def _compute_gradient(self, pattern_label):

        """
        Compute the gradients of weights and biases wrt current layer parameters during backpropagation.

        :param pattern_label: (np.array) |  The true label corresponding to the input pattern.

        :return: (tuple) | A tuple containing weights_gradient and biases_gradient arrays.
        """

        self._error = (pattern_label - self._output)
        self._sq_error.append(self._error**2)
        self._delta = self._error * self.activation_function.derivative(self._z)
        weights_gradient = np.outer(self._input, self._delta)
        biases_gradient = self._delta
        if weights_gradient.shape != self.weights.shape:
            raise ValueError("Gradient shape: {} does not match self.weights shape: {}.".format(weights_gradient.shape, self.weights.shape))
        return weights_gradient, biases_gradient


    def _forward(self):

        """
        Perform forward propagation through the OutputLayer.

        This method computes the weighted sum of inputs, applies the activation function, and sets the output.
        """

        self._z = np.dot(self._input, self.weights) + self.biases
        self._output = self.activation_function(self._z)
