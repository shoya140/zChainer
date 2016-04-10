# coding: utf-8

import numpy as np
from chainer import cuda, Variable, ChainList, optimizers, serializers
import chainer.functions as F
import os.path
import pickle
from sklearn.base import BaseEstimator
import utility

class NNManager (BaseEstimator):
    def __init__(self, model,
        optimizer=optimizers.Adam(), loss_function=F.softmax_cross_entropy,
        epoch=20, batch_size=100, log_path="", export_path="", gpu_flag=-1):
        self.model = model
        self.optimizer = optimizer
        self.optimizer.setup(self.model)
        self.loss_function = loss_function
        self.epoch = epoch
        self.batch_size = batch_size
        self.log_path = log_path
        self.export_path = export_path
        self.gpu_flag= gpu_flag

    def fit(self, x_train, y_train, x_test=[], y_test=[], is_classification=False):
        xp = np
        if self.gpu_flag >= 0:
            cuda.check_cuda_available()
            cuda.get_device(self.gpu_flag).use()
            xp = cuda.cupy
            self.model.to_gpu()

        if self.log_path != "" and not os.path.isfile(self.log_path):
            print "log_path: "+self.log_path
            utility.writeText(self.log_path, "a",
                "datetime,epoch,train_loss,test_loss,test_accuracy\n")

        for epoch in xrange(self.epoch):
            train_size = len(x_train)
            train_loss = 0.
            test_loss = 0.
            test_accuracy = 0.
            indexes = np.random.permutation(train_size)
            for i in xrange(0, train_size, self.batch_size):
                x = Variable(xp.asarray(x_train[indexes[i:i+self.batch_size]]))
                t = Variable(xp.asarray(y_train[indexes[i:i+self.batch_size]]))

                self.model.zerograds()
                loss = self.loss_function(self.forward(x), t)
                loss.backward()
                self.optimizer.update()
                train_loss += loss.data * self.batch_size
            train_loss /= train_size

            test_size = len(x_test)
            for i in xrange(0, test_size, self.batch_size):
                x = Variable(xp.asarray(x_test[i:i+self.batch_size]))
                t = Variable(xp.asarray(y_test[i:i+self.batch_size]))
                test_loss += self.loss_function(self.forward(x), t).data * self.batch_size
                if is_classification:
                    test_accuracy += F.accuracy(self.forward(x), t).data * self.batch_size
            if test_size != 0:
                test_loss /= test_size
                test_accuracy /= test_size

            if epoch > 1 and self.log_path != "":
                utility.writeText(self.log_path,"a",
                    "%s,%d,%f,%f,%f\n"% (utility.now(), epoch,
                        train_loss, test_loss, test_accuracy))

        if self.export_path != "":
            if self.gpu_flag >= 0:
                self.model.to_cpu()
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
