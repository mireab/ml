import pandas as pd
from nn.Layer import *
from nn.NeuralNetwork import *
from nn.RegressionNeuralNetwork import *
from nn.GridSearch import *

def read_monk(path):
    monktrain_path = path
    columns = ['target', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'ID']
    df = pd.read_csv(monktrain_path, names=columns, delimiter=' ', skipinitialspace=True)
    categorical_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'target']
    one_hot_encoded = pd.get_dummies(df[categorical_columns], columns= categorical_columns, prefix=categorical_columns).astype(int)
    X = one_hot_encoded.iloc[:, :-2].to_numpy()
    y = one_hot_encoded.iloc[:,-1].to_numpy()
    return X, y


def read_cup(path):
    cup_path = path
    cup_df = pd.read_csv(cup_path, skiprows=7, header=None, index_col= 0)
    columns = ['target', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'ID']
    exp_columns = ["target x", "target y", "target z"]
    ft_columns = len(cup_df.columns) - len(exp_columns)
    ft_columns = ["ft {}".format(i) for i in range(ft_columns)]
    columns = ft_columns + exp_columns
    cup_df.columns = columns
    X = cup_df.iloc[:,:-3]
    x = cup_df.iloc[:,-3]
    y = cup_df.iloc[:,-2]
    z = cup_df.iloc[:,-1]
    X, x, y, z = X.values, x.values, y.values, z.values
    return X, x, y, z

if __name__ == "__main__":
    # monk1_path =
    # monk2_path =
    # monk3_path =
    cup_path = "cup\ML-CUP23-TR.csv"
    # X, y = read_monk(monk1_path)
    X, x, y, z = read_cup(cup_path)

    parameters = {'n_units':[2, 4, 6], 
              'n_layers':[1,2], 
              'eta':[0.1, 0.05, 0.01], 
              'regularization':["tikhonov"], 
              "_lambda":[0.001, 0.005, 0.01 ], 
              'momentum':[0.0, 0.3, 0.8], 
              'activation_function':[ActivationFunctions.SIGMOID, ActivationFunctions.TANH], 
              'version': ['online'], 
              'batch_size':[None]}

    GS = GridSearch(parameters, max_iter=700, cv = True, val_split=0.1, folds=None, regression = True)
    GS.fit_and_evaluate(X, x, plot=False)
    # layers=[InputLayer(17), HiddenLayer(6), OutputLayer(1)]
    # Network=NeuralNetwork(layers = layers, learning_rate= 30, momentum = 0.5, random_seed=86)
    # Network.crossvalidation=False
    # Network.train(X, y, epochs=100, version='batch', epsilon=0.005, verbose=True, crossvalidation=False, val_split = None, plot = True, stochastic=False)