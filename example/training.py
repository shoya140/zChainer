# coding: utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers
from sklearn.base import ClassifierMixin
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
import pickle

import sys,os.path
sys.path.append('../')
from zChainer import NNManager, NNAutoEncoder, utility

mnist = fetch_mldata('MNIST original', data_home=".")
data = mnist.data.astype(np.float32)
label = mnist.target.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1)

# Create a new network
#model = ChainList(L.Linear(784, 1000), L.Linear(1000, 100), L.Linear(100, 10))

# or load a serialized model
f = open("./ae_2015-12-01_11-26-45.model")
model = pickle.load(f)
f.close()
model.add_link(L.Linear(100,10))

def forward(self, x):
    h = F.relu(self.model[0](x))
    h = F.relu(self.model[1](h))
    return F.relu(self.model[2](h))

NNManager.forward = forward
nn = NNManager(model, optimizers.Adam(), F.softmax_cross_entropy, epoch=100, batch_size=100,
    log_path="./training_"+utility.now()+"_log.csv", export_path="./training_"+utility.now()+".model")
nn.fit(X_train, y_train, X_test, y_test)
