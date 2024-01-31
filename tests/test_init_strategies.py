import numpy as np
import pytest

from nn.InitStrategies import InitStrategies, _DEFAULT_SCALE

@pytest.mark.parametrize("strategy, input_size, unit_number, expected", [
    (InitStrategies.RANDOM, 10, 5, _DEFAULT_SCALE),
    (InitStrategies.GLOROT, 10, 5, np.sqrt(6 / (10 + 5))),
    # Add more test cases if needed
])
def test_scale(strategy, input_size, unit_number, expected):
    assert strategy.scale(input_size, unit_number) == pytest.approx(expected, rel=1e-5)

@pytest.mark.parametrize("strategy, input_size, unit_number", [
    (InitStrategies.RANDOM, 10, 5),
    (InitStrategies.GLOROT, 10, 5),
    # Add more test cases if needed
])
def test_initial_weights(strategy, input_size, unit_number):
    weights = strategy.init_weights(input_size, unit_number)
    scale = strategy.scale(input_size, unit_number)
    assert weights.shape == (input_size, unit_number)
    assert np.all(weights >= -scale) and np.all(weights <= scale)

@pytest.mark.parametrize("strategy, input_size, unit_number", [
    (InitStrategies.RANDOM, 10, 5),
    (InitStrategies.GLOROT, 10, 5),
    # Add more test cases if needed
])
def test_initial_bias(strategy, input_size, unit_number):
    bias = strategy.init_biases(input_size, unit_number)
    assert len(bias) == unit_number
    if strategy == InitStrategies.RANDOM:
        assert np.all(bias >= -_DEFAULT_SCALE) and np.all(bias <= _DEFAULT_SCALE)
    elif strategy == InitStrategies.GLOROT:
        assert np.all(bias == 0)
