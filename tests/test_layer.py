import pytest
import numpy as np
from nn.InitStrategies import InitStrategies
from nn.ActivationFunctions import ActivationFunctions
from nn.Layer import InputLayer, HiddenLayer, OutputLayer

class MockLayer:
    def __init__(self, unit_number):
        self.unit_number = unit_number
        self._output = np.random.random(unit_number)
        self._delta = np.random.random(unit_number)

@pytest.fixture
def mock_input_layer():
    layer = MockLayer(3)
    layer._output = np.array([1, 2, 3])
    return layer

@pytest.fixture
def mock_next_layer():
    layer = MockLayer(3)
    layer._delta = np.array([0.3, 0.4, 0.5])
    layer.weights = np.array([[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]])
    return layer


@pytest.mark.parametrize("layer_type", [HiddenLayer, OutputLayer])
def test_forward_pass_updates_z_value_and_outputs(layer_type, mock_input_layer):
    layer = layer_type(2, activation_function=ActivationFunctions.SIGMOID, init_strategy=InitStrategies.RANDOM)
    layer._previous_layer = mock_input_layer

    # Manually set weights and biases for controlled testing
    layer.weights = np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]])
    layer.biases = np.array([2.1, 2.2])
    
    layer._forward()
    expected_z = np.array([10.3, 11])
    assert np.allclose(layer._z, expected_z)
    assert np.allclose(layer._output, ActivationFunctions.SIGMOID(expected_z))

def test_hidden_layer_computed_gradients(mock_input_layer, mock_next_layer, monkeypatch):
    hidden_layer = HiddenLayer(2, activation_function=ActivationFunctions.SIGMOID, init_strategy=InitStrategies.RANDOM)
    hidden_layer._previous_layer = mock_input_layer
    hidden_layer._next_layer = mock_next_layer

    # Manually set weights and biases for controlled testing
    hidden_layer.weights = np.array([[1.1, 1.2], [1.3,1.4], [1.5, 1.6]])
    hidden_layer.biases = np.array([2.1, 2.2])

    monkeypatch.setattr(ActivationFunctions, "derivative", lambda *args: np.array([1, 2]))
    
    hidden_layer._forward()
    print(hidden_layer._compute_gradient())
    expected_updated_weights = np.array([[3.86, 8.44],[7.72, 16.88], [11.58, 25.32]])
    expected_updated_biases = np.array([3.86, 8.44])
    updated_weights, updated_biases = hidden_layer._compute_gradient()
    assert np.allclose(updated_weights,expected_updated_weights)
    assert np.allclose(updated_biases,expected_updated_biases)

    


def test_input_layer_forwards_the_input_values_as_output():
    unit_number = 5
    input_layer = InputLayer(unit_number)
    test_input = np.array([1, 2, 3, 4, 5])

    input_layer._set_input(test_input)
    assert np.array_equal(input_layer._value, test_input)

    input_layer._forward()
    assert np.array_equal(input_layer._output, test_input)

def test_hidden_layer_shape_as_expected_on_initialisation():
    unit_number = 5
    input_size = 3
    hidden_layer = HiddenLayer(unit_number)
    hidden_layer._previous_layer = InputLayer(input_size)  # Mock previous layer
    hidden_layer._initialize_weights()

    assert hidden_layer.weights.shape == (input_size, unit_number)
    assert hidden_layer.biases.shape == (unit_number,)


def test_output_layer_shape_as_expected_on_initialisation():
    unit_number = 5
    input_size = 3
    output_layer = OutputLayer(unit_number)
    output_layer._previous_layer = InputLayer(input_size)  # Mock previous layer
    output_layer._initialize_weights()

    assert output_layer.weights.shape == (input_size, unit_number)
    assert output_layer.biases.shape == (unit_number,)



