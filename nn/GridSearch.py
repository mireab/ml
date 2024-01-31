import numpy as np
from nn.NeuralNetwork import *
from nn.Layer import *
from nn.utils import *
from itertools import product
from tqdm import tqdm 
import sys


class GridSearch:
    def __init__(self, grid, random_seed = 42, cv = False, val_split=None, max_iter = 1000, folds = None):
        self.grid = grid
        self._best_params = None
        self.metrics = None
        self.val_split = val_split
        self.max_iter = max_iter
        self.cv = cv
        self.folds = folds
        self.random_seed = random_seed

    def fit_and_evaluate(self, X, y, plot = False, ):
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
            # if not (attributes['version'] != 'minibatch' and isinstance(attributes['batch_size'], int)):
            if not (not attributes['regularization'] and attributes['_lambda'] != 0.0):
                layers = [InputLayer(len(X[0]))]
                for i in range(attributes['n_layers']):
                    layers.append(HiddenLayer(attributes['n_units'], activation_function=attributes['activation_function']))
                layers.append(OutputLayer(1, activation_function = attributes['activation_function']))

                network = NeuralNetwork(layers=layers, random_seed=self.random_seed)
                attributes_to_set_NN = {key: value for key, value in attributes.items() if key not in ["n_layers", "n_units", "activation_function", "batch_size", "version"]}
                network.set_attributes(attributes_to_set_NN)
                
                if self.cv:
                    if self.val_split is not None:
                        network.train(X, y, epochs=self.max_iter, version=attributes['version'], batch_size=attributes['batch_size'], epsilon=0.005, crossvalidation=False, val_split = self.val_split, plot = plot)
                                #, quickprop = attributes['quickprop'])
                    else:
                        network.train(X, y, epochs=self.max_iter, version=attributes['version'], batch_size=attributes['batch_size'], epsilon=0.005, crossvalidation=self.cv, val_split = None, plot = plot, n_folds = self.folds)
                                    #, quickprop = attributes['quickprop'])
                    attributes = tuple(attributes.items())
                    performances[attributes] = np.mean([fold['val_loss'][-1] for fold in network._history])
                else:
                    network.train(X, y, epochs=self.max_iter, version=attributes['version'], batch_size=attributes['batch_size'], epsilon=0.005, plot = plot)
                                # quickprop = attributes['quickprop'])
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

        min_loss = min(performances.values())
        _best_params = [combo for combo, loss in performances.items() if loss == min_loss]
        return _best_params