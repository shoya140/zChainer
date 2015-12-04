# coding: utf-8

import numpy as np
from chainer import Variable, ChainList, optimizers, serializers
import chainer.functions as F
import os.path
import pickle
from sklearn.base import BaseEstimator
import utility

class NNManager (BaseEstimator):
    def __init__(self, model,
        optimizer=optimizers.Adam(), loss_function=F.softmax_cross_entropy,
        epoch=20, batch_size=100, log_path="", export_path=""):
        self.model = model
        self.optimizer = optimizer
        self.optimizer.setup(self.model)
        self.loss_function = loss_function
        self.epoch = epoch
        self.batch_size = batch_size
        self.log_path = log_path
        self.export_path = export_path

    def fit(self, x_train, y_train, x_test=[], y_test=[], isClassification=False):
        if self.log_path != "" and not os.path.isfile(self.log_path):
            print "log_path: "+self.log_path
            utility.writeText(self.log_path, "a",
                "datetime,epoch,train_loss,test_loss,test_accuracy\n")

        for epoch in xrange(self.epoch):
            train_size = len(x_train)
            indexes = np.random.permutation(train_size)
            train_loss = 0.
            test_loss = 0.
            test_accuracy = 0.
            for i in xrange(0, train_size, self.batch_size):
                x = Variable(x_train[indexes[i:i+self.batch_size]])
                t = Variable(y_train[indexes[i:i+self.batch_size]])

                self.model.zerograds()
                loss = self.loss_function(self.forward(x), t)
                loss.backward()
                self.optimizer.update()
                train_loss += loss.data * self.batch_size
            train_loss /= train_size

            if len(x_test) != 0:
                x = Variable(x_test)
                t = Variable(y_test)
                test_loss = self.loss_function(self.forward(x), t).data
                if isClassification:
                    test_accuracy = F.accuracy(self.forward(x), t).data

            if epoch > 1 and self.log_path != "":
                utility.writeText(self.log_path,"a",
                    "%s,%d,%f,%f,%f\n"% (utility.now(), epoch,
                        train_loss, test_loss, test_accuracy))

            if self.export_path != "":
                pickle.dump(self.model, open(self.export_path, 'wb'), -1)
        return self

    def predict(self, x):
        return self.output(self.forward(Variable(x)))

    def output(self, y):
        return y.data

    def forward(self, x_data):
        raise NotImplementedError("""`forward` method is not implemented.
        example)
        h1 = F.relu(self.model[0](x))
        h2 = F.relu(self.model[1](h1))
        return F.relu(self.model[2](h2))""")
