import mxnet as mx
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import data as gdata, loss as gloss, nn
import time
import function as fun

batch_size = 16

# get data
X, Y = fun.get_data()
dataset = gdata.ArrayDataset(X,Y)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# get model
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(1))
net.initialize(init = init.Xavier())

# setup environment
epochs = 50
lr = 0.03
loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})


# start trainning
for e in range(epochs):

    total_loss = 0
    start = time.time()
    for x, y in data_iter:
        with autograd.record():
            yhat = net(x)
            l = loss(yhat, y)
        l.backward()
        trainer.step(batch_size)
        total_loss += l.sum().asscalar()

    print('epoch: %d, loss: %f, time: %f sec' % (e + 1, total_loss / len(Y), time.time() - start))
    
# show weight & bias
print(net[0].weight.data(), net[0].bias.data())

