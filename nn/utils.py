import numpy as np


def mse(y_true, y_pred):
    if not (np.array(y_true).size and np.array(y_pred).size):
        raise ValueError("Cannot calculate mse on empty-array input")
    return np.mean([(tgt - pred)**2 for tgt, pred in zip(y_true, y_pred)])

def threshold_function(value, threshold):
    return int(value >= threshold)


def pairwise_sum(tuple_list_a, tuple_list_b):
    return [(np.add(a_ws, b_ws), np.add(a_bs, b_bs)) for (a_ws, a_bs), (b_ws, b_bs) in zip(tuple_list_a, tuple_list_b)]