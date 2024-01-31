import numpy as np
from nn.NeuralNetwork import *
from nn.RegressionNeuralNetwork import *
from nn.Layer import *
from nn.utils import *
from itertools import product
from tqdm import tqdm 
import sys


class GridSearch:
    
    """
    A class for performing grid search optimization over a set of parameters for a neural network. 
    This class allows the user to specify a range of values for different hyperparameters 
    and then evaluates these combinations to find the best set of parameters based on performance metrics.

    :param grid: (dict) | A dictionary where keys are the names of parameters and values are lists 
                of parameter settings to try as grid points.
    :param random_seed: (int) | default: 42 | The seed for the random number generator.
    :param cv: (bool) | default: False | Flag to determine if cross-validation or hold-out should be used.
    :param val_split: (float) | default: None | The fraction of the dataset to use as validation set.
    :param max_iter: (int) | default: 1000 | The maximum number of iterations for training.
    :param folds: (int or None) | default: None | The number of folds to use for cross-validation.
    :param regression: (bool) | default: False | Flag to indicate if the problem is a regression problem.
    """

    def __init__(self, grid, random_seed = 42, cv = False, val_split=None, max_iter = 1000, folds = None, regression = False):
        self.grid = grid
        self._best_params = None
        self.metrics = None
        self.val_split = val_split
        self.max_iter = max_iter
        self.cv = cv
        self.folds = folds
        self.random_seed = random_seed
        self.regression = regression

    
    def _save_to_csv(self, filename, data):
        
        """
        Function to save the performance metrics of the grid search to a CSV file.

        :param filename: (str) | The name of the file to save the data.
        :param data: (dict) | A dictionary containing the combinations of parameters and their corresponding performance metrics.

        """
        with open(filename, mode='w', newline='') as csv_file:
            fieldnames = list(self.grid.keys()) + ["MSE"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for key, mse in data.items():
                # Create a dictionary with appropriate keys for CSV writing
                row_data = dict(zip(list(self.grid.keys()), key))
                row_data["MSE"] = mse
                writer.writerow(row_data)

    def fit_and_evaluate(self, X, y, plot = False, csv_filename = None ):
        
        """
        Fits and evaluates the model using the specified combinations of hyperparameters. 
        It iterates over all possible combinations, trains the model, evaluates it, and keeps track of the performance metrics.

        :param X: (array) | The input features for the model.
        :param y: (array) | The target values for the model.
        :param plot: (bool) | default: False | Flag to determine if plots should be generated during training.
        :param csv_filename: (str or None) | default: None | Optional filename to save the performance metrics of the grid search.
        
        :return: (list) | A list of the best parameter combinations based on the evaluation metric.
        """

        performances = {}
        # Check if the _output stream supports interactive display (e.g., if it's a terminal)
        use_progress_bar = sys.stdout.isatty()
        if use_progress_bar:
            progress_bar = tqdm(total=len(list(product(*self.grid.values()))))
        else:
            progress_bar = None  # Disable the progress bar
        for combo in product(*self.grid.values()):
            attributes = {}
            for (name, value) in zip(list(self.grid.keys()), combo):
                attributes[name] = value
            if attributes['version'] == 'minibatch':
                self.stochastic = True
            else:
                self.stochastic = False
            # if not (attributes['version'] != 'minibatch' and isinstance(attributes['batch_size'], int)):
            if not (not attributes['regularization'] and attributes['_lambda'] != 0.0):
                layers = [InputLayer(len(X[0]))]
                for i in range(attributes['n_layers']):
                    layers.append(HiddenLayer(attributes['n_units'], activation_function=attributes['activation_function']))
                if self.regression:
                    layers.append(OutputLayer(1, activation_function= ActivationFunctions.ID))
                else:
                    layers.append(OutputLayer(1, activation_function = attributes['activation_function']))

                if self.regression:
                    network = RegressionNeuralNetwork(layers = layers, random_seed = self.random_seed)
                else:
                    network = NeuralNetwork(layers=layers, random_seed=self.random_seed)
                attributes_to_set_NN = {key: value for key, value in attributes.items() if key not in ["n_layers", "n_units", "activation_function", "batch_size", "version", "quickprop"]}
                network.set_attributes(attributes_to_set_NN)
                
                if self.cv:
                    if self.val_split is not None:
                        network.train(X, y, epochs=self.max_iter, version=attributes['version'], batch_size=attributes['batch_size'], epsilon=0.005, crossvalidation=False, val_split = self.val_split, plot = plot, quickprop = attributes['quickprop'], stochastic=self.stochastic)
                    else:
                        network.train(X, y, epochs=self.max_iter, version=attributes['version'], batch_size=attributes['batch_size'], epsilon=0.005, crossvalidation=self.cv, val_split = None, plot = plot, n_folds = self.folds, quickprop = attributes['quickprop'], stochastic=self.stochastic)
                    attributes = tuple(attributes.items())
                    performances[attributes] = np.mean([fold['val_loss'][-1] for fold in network._history])
                else:
                    network.train(X, y, epochs=self.max_iter, version=attributes['version'], batch_size=attributes['batch_size'], epsilon=0.005, plot = plot, quickprop = attributes['quickprop'], stochastic=self.stochastic)
                    attributes = tuple(attributes.items())
                    performances[attributes] = network._history[0]['train_loss'][-1]
                    
                p = [p for p in performances.values()]
                arg = [arg for arg in performances.items()]
                print(arg[-1], p[-1])
                del network

            if use_progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix(best_loss=min(performances.values()))

        if use_progress_bar:
            progress_bar.close()

        if csv_filename:
            self._save_to_csv(csv_filename, performances)

        min_loss = min(performances.values())
        _best_params = [combo for combo, loss in performances.items() if loss == min_loss]
        return _best_params
