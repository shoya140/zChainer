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

    def fit(self, x_train, y_train, x_test=[], y_test=[], autoencoding=False):
        if self.log_path != "" and not os.path.isfile(self.log_path):
            print "log_path: "+self.log_path
            utility.writeText(self.log_path, "a",
                "datetime,epoch,train_loss,train_acc,test_acc\n")

        for epoch in xrange(self.epoch):
            train_size = len(x_train)
            indexes = np.random.permutation(train_size)
            train_loss = 0.
            train_accuracy = 0.
            test_accuracy = 0.
            for i in xrange(0, train_size, self.batch_size):
                self.optimizer.zero_grads()

                x_batch = x_train[indexes[i:i+self.batch_size]]
                y_batch = y_train[indexes[i:i+self.batch_size]]
                y = self.forward(Variable(x_batch))

                loss = self.loss_function(y, Variable(y_batch))
                loss.backward()
                self.optimizer.update()
                train_loss += loss.data * self.batch_size
            train_loss /= train_size

            if not autoencoding:
                y_predict = self.predict(x_train)
                mx, acc = utility.confusionMatrix(y_train, y_predict)
                train_accuracy = acc

            if len(x_test) != 0:
                y_predict = self.predict(x_test)
                mx, acc = utility.confusionMatrix(y_test, y_predict)
                test_accuracy = acc

            if self.log_path != "":
                utility.writeText(self.log_path,"a",
                    "%s,%d,%f,%f,%f\n"% (utility.now(), epoch+1,
                        train_loss, train_accuracy, test_accuracy))

            if self.export_path != "":
                pickle.dump(self.model, open(self.export_path, 'wb'), -1)
        return self

    def predict(self, x_test):
        output = self.forward(Variable(x_test))
        y_trimed = output.data.argmax(axis=1)
        return np.array(y_trimed, dtype=np.int32)

    def forward(self, x_data):
        raise NotImplementedError("""`forward` method is not implemented.
        example)
        h1 = F.relu(self.model[0](x))
        h2 = F.relu(self.model[1](h1))
        return F.relu(self.model[2](h2))""")
