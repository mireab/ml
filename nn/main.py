import pandas as pd
from nn.Layer import *
from nn.NeuralNetwork import *



monk1train_path = "monk+s+problems/monks-1.train"
columns = ['target', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'ID']
df = pd.read_csv(monk1train_path, names=columns, delimiter=' ', skipinitialspace=True)
categorical_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'target']
one_hot_encoded = pd.get_dummies(df[categorical_columns], columns= categorical_columns, prefix=categorical_columns).astype(int)

X = one_hot_encoded.iloc[:, :-2].to_numpy()
y = one_hot_encoded.iloc[:,-1].to_numpy()

layers=[InputLayer(17), HiddenLayer(6), OutputLayer(1)]
Network=NeuralNetwork(layers = layers, learning_rate= 30, momentum = 0.5, random_seed=86)
Network.crossvalidation=False
Network.train(X, y, epochs=100, version='batch', epsilon=0.005, verbose=True, crossvalidation=False, val_split = None, plot = True, stochastic=False)