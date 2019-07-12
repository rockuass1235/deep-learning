import mxnet as mx
from mxnet import gluon, nd, autograd, init
from mxnet.gluon import data as gdata, loss as gloss, nn
import function as fun
import time

mnist_train = gdata.vision.MNIST(train=True)
mnist_test = gdata.vision.MNIST(train=False)

x,y = mnist_train[0:9]
#fun.show_data(x, y)


batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),batch_size, shuffle=True,)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),batch_size, shuffle=False)


net = nn.Sequential()
net.add(nn.Dense(256, activation = 'relu'))
net.add(nn.Dense(256 , activation= 'relu'))
net.add(nn.Dense(10))
net.initialize(init = init.Xavier())


epochs = 50
lr = 0.03
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

for e in range(epochs):

    total_loss = 0
    start = time.time()
    count = 0
    for x, y in train_iter:
        with autograd.record():
            yhat = net(x)
            l = loss(yhat, y)
        l.backward()
        trainer.step(batch_size)
        total_loss += l.sum().asscalar()
        count += len(y)

    print('epoch: %d, loss: %f, time: %f sec' % (e + 1, total_loss /count, time.time() - start
