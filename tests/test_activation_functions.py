
import numpy as np
import pytest

from nn.ActivationFunctions import ActivationFunctions

@pytest.mark.parametrize("function, z, expected", [
    (ActivationFunctions.SIGMOID, 0, 0.5),
    (ActivationFunctions.SIGMOID, 100, pytest.approx(1.0, rel=1e-5)),
    (ActivationFunctions.SIGMOID, -100, pytest.approx(0.0, rel=1e-5)),
    (ActivationFunctions.TANH, 0, 0),
    (ActivationFunctions.TANH, 100, pytest.approx(1.0, rel=1e-5)),
    (ActivationFunctions.TANH, -100, pytest.approx(-1.0, rel=1e-5)),
])
def test_activation_function_call(function, z, expected):
    assert function(z) == expected

@pytest.mark.parametrize("function, z, expected_range", [
    (ActivationFunctions.SIGMOID, 0, (0, 0.25)),
    (ActivationFunctions.SIGMOID, 2, (0, 0.25)),
    (ActivationFunctions.TANH, 0, (0, 1)),
    (ActivationFunctions.TANH, 2, (0, 1)),
])
def test_activation_function_derivative(function, z, expected_range):
    result = function.derivative(z)
    assert expected_range[0] <= result <= expected_range[1]
