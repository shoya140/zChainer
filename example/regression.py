# coding: utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers
from sklearn.cross_validation import train_test_split
import pickle

import sys,os.path
sys.path.append('../')
from zChainer import NNManager, NNAutoEncoder, utility

data_size = 1000
X = np.linspace(-1,1,data_size).reshape(data_size, 1).astype(np.float32)
y = np.array([2*x*x-1 for x in X], np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = ChainList(L.Linear(1, 1024), L.Linear(1024, 1))

def forward(self, x):
    h = F.relu(self.model[0](x))
    return self.model[1](h)

NNManager.forward = forward
nn = NNManager(model, optimizers.Adam(),
  F.mean_squared_error, epoch=200, batch_size=100,
  log_path="./log_regression_"+utility.now()+"_log.csv")

nn.fit(X_train, y_train, X_test, y_test)
nn.predict(X_test)
