# coding: utf-8

import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix

def now():
    return dt.now().strftime("%Y-%m-%d_%H-%M-%S")

def writeText(path, option, text):
    f = open(path, option)
    f.write(text)
    f.close()

def confusionMatrix(label, predicted):
    mx = confusion_matrix(label, predicted)
    acc = float(np.sum([mx[i][i] for i in range(len(mx))]))/float(len(label))
    return mx, acc