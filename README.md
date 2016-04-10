# zChainer

scikit-learn like interface and stacked autoencoder for chainer

## Requirements

* numpy
* scikit-learn
* chainer >= 1.5

## Installation

    pip install zChainer

## Usage

### Autoencoder

```python
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers
from zChainer import NNAutoEncoder, utility

data = (..).astype(np.float32)

encoder = ChainList(
    L.Linear(784, 200),
    L.Linear(200, 100))
decoder =ChainList(
    L.Linear(200, 784),
    L.Linear(100, 200))

# You can set your own forward function. Default is as below.
#def forward(self, x):
#    h = F.dropout(F.relu(self.model[0](x)))
#    return F.dropout(F.relu(self.model[1](h)))
#
#NNAutoEncoder.forward = forward
ae = NNAutoEncoder(encoder, decoder, optimizers.Adam(), epoch=100, batch_size=100,
    log_path="./ae_log_"+utility.now()+".csv", export_path="./ae_"+utility.now()+".model")

ae.fit(data)
```

### Training and Testing

```python
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import ChainList, optimizers
from zChainer import NNManager, utility
import pickle

X_train = (..).astype(np.float32)
y_train = (..).astype(np.int32)
X_test = (..).astype(np.float32)
y_test = (..).astype(np.int32)

# Create a new network
model = ChainList(L.Linear(784, 200), L.Linear(200, 100), L.Linear(100, 10))

# or load a serialized model
#f = open("./ae_2015-12-01_11-26-45.model")
#model = pickle.load(f)
#f.close()
#model.add_link(L.Linear(100,10))

def forward(self, x):
    h = F.relu(self.model[0](x))
    h = F.relu(self.model[1](h))
    return F.relu(self.model[2](h))

def output(self, y):
    y_trimed = y.data.argmax(axis=1)
    return np.array(y_trimed, dtype=np.int32)

NNManager.forward = forward
NNManager.output = output
nn = NNManager(model, optimizers.Adam(), F.softmax_cross_entropy,
    epoch=100, batch_size=100,
    log_path="./training_log_"+utility.now()+".csv")

nn.fit(X_train, y_train, is_classification=True)
nn.predict(X_test, y_test)
```
