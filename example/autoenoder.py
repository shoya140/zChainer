# coding: utf-8

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers
from sklearn.base import ClassifierMixin
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata

import sys,os.path
sys.path.append('../')
from zChainer import NNManager, NNAutoEncoder, utility

mnist = fetch_mldata('MNIST original', data_home=".")
data = mnist.data.astype(np.float32)

encoder = ChainList(
    L.Linear(784, 1000),
    L.Linear(1000, 100))
decoder =(
    L.Linear(1000, 784),
    L.Linear(100, 1000))
ae = NNAutoEncoder(encoder, decoder, optimizers.Adam(), epoch=2, batch_size=100,
    log_path="./ae_"+utility.now()+"_log.csv", export_path="./ae_"+utility.now()+".model")
ae.fit(data)