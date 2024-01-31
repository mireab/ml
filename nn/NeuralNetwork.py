# layer_sizes should be a vector of numbers
from random import shuffle
from matplotlib import pyplot as plt
from math import floor, exp
import numpy as np
import warnings
from nn.Layer import *
from nn.utils import *


class NeuralNetwork:

    """
    Class representing a Neural Network.

    Attributes:
        layers: The input, hidden and output layer of the NN, all objects of the Layer() class

        learning_rate : (float) | default = 0.05 | The learning rate, or eta

        regularization : (str) | default = None |  If different from None, the weights are updated 
                        following the specified regularization strategy (for this implementation, the 
                        only regularization technique is tikhonov)

        _lambda : (float) | default = None | The value for tikhonov's lambda

        momentum : (float) | default = 0.0 | If different from 0.0 in the updated of the weights the 
                    product between momentum and the previous weights update is added to the Loss
                
        random_seed : (int) | default = None | The random seed for the weights' initialitation for each layer

        loss_function : (str) | default = mse | The function used to calculate the loss: in this implementation
                        must be Mean Squared Error

    """
        
    def __init__(self, layers = None, learning_rate = 0.05, regularization = None, _lambda = None, momentum = 0.0,
                 random_seed = None, loss_function = mse):
        self.layers = layers or [InputLayer(1), OutputLayer(1)]
        self.batch_size = None  # ?????????

        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number, but received: {}"
                             .format(learning_rate))
        self.eta = learning_rate

        if random_seed is not None and not isinstance(random_seed, int):
            raise ValueError("Random seed must be an integer, but received: {}"
                             .format(random_seed))
        self.random_seed = random_seed

        self._connect_layers()
        self._initialize_network_weights()
        self._allowed_regularizations = {"tikhonov"}
        if regularization is not None and regularization not in self._allowed_regularizations:
            raise ValueError("Invalid regularization strategy: {}. Please, choose one between: {}"
                             .format(regularization, self._allowed_regularizations))

        self.regularization = regularization

        if _lambda is not None and (not isinstance(_lambda, float) or _lambda < 0):
            raise ValueError("Regularization coefficient must be a positive number, but received: {}"
                             .format( _lambda))
        self._lambda = _lambda # controllare underscore

        if not self.regularization:
            self._lambda = 0.0

        if not isinstance(momentum, float) or momentum < 0:
            raise ValueError("Momentum coefficient must be a positive number, but received: {}"
                             .format(momentum))
        self.momentum = momentum
        self.iteration = 0
        self._history = {"loss": [], "accuracy": [], 'prediction': [], 'parameters' : []}
        
        if loss_function != mse:
             raise ValueError("Only mse was implemented as loss function")
        
        self.loss_function = loss_function
        self._allowed_metrics = {'accuracy', 'loss', 'f1-score', 'recall'}
        self.threshold_value = self.layers[-1].threshold_value


    def set_attributes(self, attribute_dict):
    
        """
        Function to set the attributes of the class when given a dictionary of attributes.

        :param attribute_dict: (dict) the dictionary from which the attributes must be taken
        """
        
        for key, value in attribute_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


    def get_attributes(self):
        return self.__dict__


    @property
    def _output_layer(self):
        return self.layers[-1]

    def _initialize_network_weights(self):

        """
        Function that initializes the layers' weights.
        """

        self._trainable_layers = []
        for layer in self.layers:
            if not isinstance(layer, InputLayer):
                if self.random_seed is not None:
                    layer.random_seed = self.random_seed
                layer._initialize_weights()
                self._trainable_layers.append(layer)


    def restart (self):
        self._initialize_network_weights()

    
    def _connect_layers(self): 

        """
        Function that connects the layers by creating, for each layer, the _next_layer and _previous_layer
        attributes
        """
        for i, layer in enumerate(self.layers):
            if (isinstance(layer, InputLayer)):
                layer._next_layer = self.layers[i + 1]
            elif (isinstance(layer, HiddenLayer)):
                layer._previous_layer = self.layers[i - 1]
                layer._next_layer = self.layers[i + 1]
            elif (isinstance(layer, OutputLayer)):
                layer._previous_layer = self.layers[i - 1]


    def _forward_propagation(self, input_pattern):

        """
        Function that, given an input pattern, returns the prediction for that pattern, using the 
        .forward() method for each layer, performing a forward propagation.

        :param input_pattern: (array of shape (1, number of features)) | The singular sample from X to predict

        :return: (float) | A single prediction 
        """

        self.layers[0]._set_input(input_pattern)
        for layer in self.layers:
            layer._forward()
        prediction = self.layers[-1]._output
        return prediction
    
    
    def predict(self, input_patterns):

        """
        Function that, given the input patterns (X), returns a list of size (number of samples) with the 
        prediction for each pattern.

        :param input_patterns: (array of shape (number of samples, number of features)) | The array of X's samples

        :return: (list of size (number of samples)) | A list of predictions
        """
        
        if type(input_patterns) == list:
            input_patterns = np.array(input_patterns)
        predictions=[]
        for pattern in input_patterns:
            assert isinstance(pattern, np.ndarray), "Each input pattern must be a numpy array"
            assert pattern.shape[0] == self.layers[0].unit_number, ("Input pattern has incorrect number of features. \
                                                                    Expected {}, got {}").format(self.layers[0].unit_number, pattern.shape[0])
            prediction = self._forward_propagation(pattern)
            predictions.append(prediction)
        return predictions
    
    def time_lr_decay (self, decay):
        self.eta *= 1/ (1 + decay/(self.iteration + 1))

    def step_decay (self, initial_rate, drop, epochs_drop):
        self.eta = initial_rate * (drop ^ (floor((1 +self.iteration) / epochs_drop)))

    def exponential_decay(self, decay):
        self.eta *= exp(-decay * (1 + self.iteration))
    
    
    def _update_weights(self, cumulative_gradients):
                
        """
        Function that, given the sum of the gradients for each pattern, updates the weights of each layer in the NN
        applying (if required) the momentum, the regularization and eventually performing quick-prop instead of
        back-prop and scaling eta adaptatively using adagrad algorithm

        :param cumulative_gradients: (array of shape (number of layers, number of samples)) | A list containing a list 
        for each layer with the sum of the layer's gradients for each pattern.
        """
        if self.eta_decay is not None:
            self.eta_decay(*self.decay_args)
        epsilon = 1e-8 # to avoid divisions with 0 denominator 
        for (weights_gradient, biases_gradient), layer in zip(cumulative_gradients,reversed(self._trainable_layers)):
            weights_difference = None
            bias_difference = None
            if self.adagrad:
                if layer.Gt.shape != weights_gradient.shape:
                    layer.Gt = layer.Gt.reshape(-1, 1)  # Reshape Gt from (5,) to (5,1)
                    layer.bias_Gt = layer.bias_Gt.reshape(-1, 1)
                layer.Gt += weights_gradient ** 2
                layer.bias_Gt += biases_gradient ** 2
                weights_difference = self.eta / self.batch_size * (weights_gradient / (1e-8 + np.sqrt(layer.Gt)))
                bias_difference = self.eta / self.batch_size * (biases_gradient / (1e-8 + np.sqrt(layer.bias_Gt)))

            if self._quickprop and layer._previous_w_gradient is not None:
                denominator_w = (layer._previous_w_gradient - weights_gradient) + epsilon
                weights_difference = self.eta/self.batch_size * (layer._previous_weights_update * (weights_gradient / denominator_w))
                denominator_b = (layer._previous_b_gradient - biases_gradient) + epsilon
                bias_difference = self.eta/self.batch_size * (layer._previous_bias_update * (biases_gradient / denominator_b))
            
            else:
                weights_difference = self.eta/self.batch_size * weights_gradient
                bias_difference = self.eta/self.batch_size * biases_gradient
            
            weights_difference += (self.momentum * layer._previous_weights_update) # To store the recent update
            
            if self.regularization=='tikhonov':
                weights_difference -= self._lambda * layer.weights
            layer._previous_weights_update = weights_difference
            layer._previous_bias_update = bias_difference
            layer.weights += weights_difference
            layer.biases += bias_difference
            layer._previous_w_gradient = weights_gradient
            layer._previous_b_gradient = biases_gradient
        self.iteration += 1

    
    def _compute_gradients(self, target):

        """
        Function that, given a target label, computes the gradient for each layer, using its ._compute_gradient() method

        :param target: (int, either 0 or 1) | the target label for the input pattern

        :return: (list of size(number of layers)) | A list containing a list for each layer with its gradient
        """

        # Abbiamo, di un solo pattern: a: [(w1, b1), (w2, b2), (w3, b3)]
                                    #  b: [(w1, b1), (w2, b2), (w3, b3)]
        # Returns  [(a:w1 + b:w1, a:b1 + b:b1), (a:w2 + b:w2, a:b2 + b:b2), ...]

        return [l._compute_gradient(target) for l in reversed(self._trainable_layers)]
  

    def _back_forw_propagation(self, X_mini_batch, y_mini_batch): 
        """
        Function that, given an array of training patterns and an array of target labels, for each tuple (pattern, target) 
        performs a forward pass and then creates a list that contains a list for each layer with the sum of its gradients 
        for every training pattern. It then uses this list to update the weights and biases.

        :param X_mini_batch: (array of shape (batch size, number of features)) | Array of training patterns
        :param y_mini_batch: (array of size (batch size)) | Array of target labels

        :return: (float) | The squared error of the output layer
        """

        cumulative_gradient = None
        for pattern, target in zip(X_mini_batch, y_mini_batch):
            self._forward_propagation(pattern)
            pattern_gradient = self._compute_gradients(target)
            cumulative_gradient = pairwise_sum(cumulative_gradient, pattern_gradient) if cumulative_gradient else pattern_gradient
        self._update_weights(cumulative_gradient)
        return self._output_layer._sq_error
        

    def metrics(self, prediction, y_test, metric=""):

        """
        This function, given a list of predictions and a list of target labels, calculates the given metric.

        :param prediction: (array of size (batch size)) | Array of predictions from the network
        :param y_test: (array of size(batch size)) | Array of target labels from the dataset
        :param metric: (str) | The desired metric to calculate. In this implementation can be 'accuracy or 'loss'

        :return: (float) | the desired metrics
        """

        if len(prediction) != len(y_test):
            raise ValueError("Length mismatch between pred and y_test.")

        thresholded = [self.threshold_function(pred) for pred in prediction]
        correct_predictions = (thresholded == y_test).sum()
        accuracy = correct_predictions / len(y_test)
        loss = self.loss_function(y_test, prediction)
        if metric == "accuracy":
            return accuracy
        if metric == "loss":
            return loss
        else:
            return accuracy, loss


    def compute_loss(self, prediction, y):
        if len(prediction) != len(y):
            raise ValueError("Length mismatch between prediction and y_test.")
        return self.loss_function(y, prediction)
    

    def compute_accuracy(self, prediction, y):
        if len(prediction) != len(y):
            raise ValueError("Length mismatch between prediction and y_test.")
        thresholded = [self.threshold_function(pred) for pred in prediction]
        correct_predictions = (thresholded == y).sum()
        accuracy = correct_predictions / len(y)
        return accuracy


    def classification_report(self, prediction, y): # Potremmo anche chiamarla confusion matrix 
        
        """
        Function that, given a list of predictions and a list of target labels, creates the classification report with precision,
        recall, f1-score and accuracy

        :param prediction: (array of size (batch size)) | Array of predictions from the network
        :param y: (array of size(batch size)) | Array of true labels from the dataset

        :return: (dict of size(4)) | A dictionary with precision, recall, accuracy and f1-score
        """
        
        if len(prediction) != len(y):
            raise ValueError("Length mismatch between prediction and y_test.")
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        report = {}

        for pred, true_label in zip(prediction, y):
            if pred == 0 and true_label == 0:
                tn += 1
            elif pred == 1 and true_label == 1:
                tp += 1
            elif pred == 0 and true_label == 1:
                fn += 1
            elif pred == 1 and true_label == 0:
                fp += 1

        report["precision"] = tp / (tp + fp) if tp + fp != 0 else 0
        report["recall"] = tp / (tp + fn) if tp + fn != 0 else 0
        report["accuracy"] = (tp + tn) / len(y) if len(y) != 0 else 0
        if report["precision"] + report["recall"] != 0:
            report["f1_score"] = 2 * report["precision"] * report["recall"] / (report["precision"] + report["recall"])
        else:
            report["f1_score"] = 0

        return report


    def threshold_function(self, value):

        """
        Function that, given a values, returns 1 if the value is >= than the threshold stored in self.threshold_value
        and 0 otherwise.

        :param value: (float) | number that needs to be thresholded

        :return: (int) | 1 or 0
        """

        if value >= self.threshold_value:
            return 1
        else:
            return 0
        
    def split_data_validation(self, X, y):

        """
        Function that, given an array of patterns X and an array of labels y, splits them following the validation method
        selected. 
        When k-fold crossvalidation is required, it will shuffle the data and then split them in the folds specified in the 
        self.n_folds param. It will then iteratively use one of the folds as the validation dataset.
        If we want to perform a simple hold-out validation it will shuffle the data and then split them according to the 
        self.val_split parameter, that represents the percentage of the dataset that should be use for validation (i.e. 0.2 = 20%).

        :param X: (array of shape (batch size, number of features)) | Array of training patterns
        :param y: (array of size(batch_size)) | Array of target labels 

        :return X_train: if k_fold cv: (array of size (n_folds) |  Array that contains the (n_folds) arrays of shape 
                                        (batch_size - batch_size / n_folds, number of features) of training patterns
                        if holdout: (array of shape (batch size - val_split * batch_size, number of features)) | Array of training 
                                    patterns
        :return X_val: if k_fold cv: (array of size (n_folds) | Array that contains the (n_folds) arrays of shape (batch_size/n_folds, 
                                    number of features) of validation patterns
                        if holdout: (array of shape (val_split * batch_size, number of features)) | Array of validation patterns
        :return y_train: if k_fold cv: (array of size (n_folds)) | Array of (n_folds) arrays of size (batch_size - batch_size/n_folds)
                                        of target labels for the training patterns
                        if holdout: (array of size (batch_size - val_split * batch_size)) | Array of target labels for the training patterns
        :return y_val: if k_fold cv: (array of size (n_folds)) | Array of (n_folds) arrays of size (batch_size/n_folds) of target labels 
                                    for the validation patterns
                        if holdout: (array of size (val_split * batch_size)) | Array of target labels for the validation patterns

        """

        if self.n_folds != 0 and self.val_split is not None or self.n_folds is None and self.val_split is None:
            raise ValueError("Provide either 'n_folds' or 'val_split', but not both.")
        if self.n_folds != 0 and self.crossvalidation == False:
            raise ValueError("In order to perform crossvalidation set the 'crossvalidation' argument as True and provide a 'n_folds'")
        

        if self.crossvalidation == True:
            X_train = []
            X_val = []
            y_train = []
            y_val = []
            if type(self.n_folds) != int:
                raise ValueError("The parameter 'n_folds' must be an integer")
            indices = np.arange(len(y))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            Xfolds = np.array_split(X, self.n_folds)
            yfolds = np.array_split(y, self.n_folds)
            for i in range(self.n_folds):
                tr_data, test_data = np.concatenate(Xfolds[:i] + Xfolds[i + 1:]), Xfolds[i]
                tr_labels, test_labels = np.concatenate(yfolds[:i] + yfolds[i + 1:]), yfolds[i]
                X_train.append(tr_data)
                X_val.append(test_data)
                y_train.append(tr_labels) 
                y_val.append(test_labels)
        
        elif self.val_split is not None:
            if self.val_split <= 0 or self.val_split >= 1:
                raise ValueError("val_split should be in the range (0, 1)")
            data_len = len(X)
            val_size = int(data_len * self.val_split)
            indices = np.arange(data_len)
            np.random.shuffle(indices)
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            X_train = X[train_indices]
            X_val = X[val_indices]
            y_train = y[train_indices]
            y_val = y[val_indices]
            
        return X_train, X_val, y_train, y_val
    
    
    def fit(self, X, y, version, verbose, stochastic, epoch, batch_size, fold=None):
    
        """
        
        """
   
        
        if version == "minibatch":
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("Batch size must be a positive integer for minibatch training. Got {} instead.".format(batch_size))
            self.batch_size = batch_size
        if version == "online":
            self.batch_size = 1
        if version not in ['minibatch', 'batch', 'online']:
            warnings.warn("The version you set is unavailable. By default, it will be used the online version")
        if version == "batch":
            self.batch_size = len(y)
        epoch_loss = [] # Resetto la loss sulla singola epoca
        batch_losses = [] # Reset loss sui batch
        prediction = []
        n_batches = len(y)//self.batch_size 
        remaining_patterns = len(y) % self.batch_size

        if stochastic:
            indices = np.arange(len(y))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
        for batch_idx in range(n_batches):
            self.layers[-1]._sq_error = [] # Ad ogni batch resetto _sq_error
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            y_mini_batch = y[start_idx:end_idx] # get the current mini-batch for y
            X_mini_batch = X[start_idx:end_idx] # get the current mini-batch for X
            batch_losses.append(self._back_forw_propagation(X_mini_batch, y_mini_batch))
            prediction.extend(self.predict(X_mini_batch))

            if verbose:
                print(f"Minibatch {batch_idx} ----- Loss: {np.mean(batch_losses[batch_idx])}")

        if remaining_patterns > 0:
            self.layers[-1]._sq_error = [] # Ad ogni epoca resetto _sq_error
            y_mini_batch = y[-remaining_patterns:]
            X_mini_batch = X[-remaining_patterns:]
            batch_losses.append(self._back_forw_propagation(X_mini_batch, y_mini_batch))
            prediction.extend(self.predict(X_mini_batch))
            
            if verbose:
                print(f"Minibatch {n_batches} ----- Loss: {np.mean(batch_losses[n_batches])}")

        epoch_loss = np.mean([sum(batch_losses, [])])
        prediction = [self.threshold_function(value) for value in prediction]
        e_report = self.classification_report(prediction, y)
        if verbose:
            print(f"Epoch {epoch}/{self.epochs} ----- Training Loss: {epoch_loss} - Training Accuracy: {e_report['accuracy']}")
            
        self._history[fold]["train_loss"].append(epoch_loss) 
        self._history[fold]["train_prediction"].append(prediction)
        self._history[fold]["train_accuracy"].append(e_report['accuracy'])
        return epoch_loss


    def evaluate(self, X_val, y_val, verbose, epoch, fold = None):
        val_pred = self.predict(X_val)
        val_predictions = [self.threshold_function(value) for value in val_pred]
        val_loss = self.loss_function(y_val, val_pred)
        e_val_report = self.classification_report(val_predictions, y_val)
        self._history[fold]["val_loss"].append(val_loss)
        self._history[fold]["val_prediction"].append(val_predictions)
        self._history[fold]["val_accuracy"].append(e_val_report['accuracy'])
        if verbose:
            print(f"Epoch {epoch}/{self.epochs} - Fold {fold} ----- Validation Loss: {val_loss} - Validation Accuracy: {e_val_report['accuracy']}")

    
    def plot_fold_losses(self,history):
        for i, fold_data in enumerate(history):
            fig, ax = plt.subplots()
            train_loss = fold_data['train_loss']
            val_loss = fold_data['val_loss']
            ax.plot(range(len(train_loss)), train_loss, label=f'Fold {i+1} Training Loss')
            ax.plot(range(len(val_loss)), val_loss, label=f'Fold {i+1} Validation Loss')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training and Validation Loss - Fold {i+1}")
            ax.legend()
            plt.show()

    

    def train(self, X, y, epochs, version,  epsilon, verbose = False, batch_size=None, stochastic = None, crossvalidation = False, n_folds = 0, val_split = None, plot = False, eta_decay = None, decay_args = None, quickprop = False):
        self.crossvalidation = crossvalidation
        self._quickprop = quickprop 
        self.n_folds = n_folds
        self.val_split = val_split
        self.epochs = epochs
        self.eta_decay = eta_decay
        self.decay_args = decay_args
        converged = False
        train_losses = []
        if self.crossvalidation:
            self._history = [{
                "train_loss": [],
                "train_prediction": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_prediction": [],
                "val_accuracy": []
            } for fold in range(self.n_folds)]
        else:
            self._history = [{
                "train_loss": [],
                "train_prediction": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_prediction": [],
                "val_accuracy": []
            }]

        if not isinstance(epsilon, float) or epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon must be a positive float between 0 and 1.Got {} instead.".format(epsilon))

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer. Got {} instead.".format(epochs))

        if len(X) != len(y):
            raise ValueError("Input X and y must have the same number of patterns.")

        # we split the data according to the validation method explicited in the parameters (if there is one)
        if crossvalidation == True:

            X_train, X_val, y_train, y_val = self.split_data_validation(X, y)

        elif val_split is not None:
            X_train, X_val, y_train, y_val = self.split_data_validation(X, y)
 
        else:
            X_train = X
            y_train = y

        e = 0 
        if self.crossvalidation:
            for fold in range(self.n_folds):
                self.restart()
                fold_losses = []
                e = 0
                while not converged and e < epochs:
                    e_loss = self.fit(X_train[fold], y_train[fold], version=version, verbose = verbose, stochastic = stochastic, epoch = e, fold=fold, batch_size = batch_size)
                    fold_losses.append(e_loss)
                    self.evaluate(X_val[fold], y_val[fold], verbose = verbose, epoch = e, fold = fold)
                    if e > 0 and abs(fold_losses[e]) < epsilon:
                        converged = True
                        print(f"Converged at epoch {e}.")
                    e += 1  # Increment the epoch counter
                self._history[fold]["train_loss"].append(fold_losses)

        else:
                while not converged and e < epochs:
                    e_loss = self.fit(X_train, y_train, version=version, verbose = verbose, stochastic = stochastic, epoch = e, fold = self.n_folds, batch_size = batch_size)
                    train_losses.append(e_loss)
                    if val_split is not None:
                        self.evaluate(X_val, y_val, verbose = verbose, epoch = e, fold = self.n_folds)
                    else:
                        self.evaluate(X_train, y_train, verbose = verbose, epoch = e, fold = self.n_folds)
                    if e > 0 and abs(train_losses[e]) < epsilon:
                        converged = True
                        print(f"Converged at epoch {e}.")
                    e += 1  # Increment the epoch counter
                self._history[0]["train_loss"]=train_losses

        if plot:
            self.plot_fold_losses(self._history)
                
        if not converged:
             print("Convergence not reached within the specified threshold.")
        # print(len(self._history["train_loss"]))
        return self._history
    

    @property
    def training_accuracy(self):
        assert len(self._history['accuracy'])!=0, "Network has not been trained."
        return self._history['accuracy'][-1]
    
    @property
    def training_loss(self):
        assert len(self._history['loss'])!=0, "Network has not been trained."
        return self._history['loss'][-1]

    def training_curve(self):
        plt.plot(range(len(self._history["loss"])), self._history["loss"], marker=',', ls="")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")