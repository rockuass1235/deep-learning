import pandas as pd
import mxnet as mx
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import nn, data as gdata, loss as gloss
import time


batch_size = 128
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
num_idx = features.dtypes[features.dtypes != 'object'].index
features[num_idx] = features[num_idx].apply(lambda x: (x - x.mean()) / (x.std()))
features[num_idx] = features[num_idx].fillna(0)
features = pd.get_dummies(features, dummy_na=True)
n_train = train_data.shape[0]
train_features = nd.array(features[:n_train].values)
test_features = nd.array(features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)


loss = gloss.L2Loss()

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation= 'tanh'))
    net.add(nn.Dense(128, activation= 'tanh'))
    net.add(nn.Dense(1))
net.initialize(init = init.Xavier())


epochs = 50000
lr = 0.03
decay = 0.001

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': decay})



for e in range(epochs):
    total_loss = 0
    start = time.time()
    for x, y in train_iter:
        with autograd.record():
            l = loss(net(x), y)
        l.backward()
        trainer.step(batch_size)
        total_loss += l.sum().asscalar()

    print('epoch: %d, loss: %f, time: %f sec' %(e+1, total_loss/len(train_data), time.time()-start))
