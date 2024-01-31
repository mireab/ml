import pytest
import numpy as np

from nn.utils import threshold_function, mse


@pytest.mark.parametrize("value, threshold, expected", [
    (10, 0, 1),
    (10, 20, 0),
])
def test_threshold_function(value, threshold, expected):
    assert threshold_function(value, threshold) == expected


@pytest.mark.parametrize("y_true, y_pred, expected", [
    (np.array([1, 2, 3]), np.array([1, 2, 3]), 0),                        # Identical arrays
    (np.array([1, 2, 3]), np.array([2, 3, 4]), 1),                        # Constant difference
    (np.array([0, 0, 0]), np.array([1, 1, 1]), 1),                        # One array all zeros
    (np.array([1000, 2000, 3000]), np.array([1000, 2000, 3000]), 0),      # Large numbers, identical arrays
    (np.array([1000, 2000, 3000]), np.array([3000, 2000, 1000]), 2666666.6666666666), # Large numbers, different arrays
])
def test_mse(y_true, y_pred, expected):
    np.testing.assert_almost_equal(mse(y_true, y_pred), expected, decimal=5)

@pytest.mark.parametrize("y_true, y_pred", [
    (np.array([1, 2, 3]), np.array([])),
    (np.array([]), np.array([1, 2, 3])),

])
def test_mse_throws_error_on_empty_array(y_true, y_pred):
    with pytest.raises(ValueError) as error:
        mse(y_true, y_pred)
        assert error.msg == "Cannot calculate mse on empty-array input"