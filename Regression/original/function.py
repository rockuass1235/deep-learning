
import mxnet as mx
from mxnet import nd, autograd
from matplotlib import pyplot as plt
import random
import time
import model as md



def get_data(dims = 2, num = 1000):

    true_w = nd.array([2, -3.4])
    true_b = 4.2

    x = nd.random.normal(shape = (num, dims))
    y = (x * true_w).sum(axis = 1) + true_b
    y = y.reshape((-1,1))

    return x, y + nd.random.normal(scale=0.01, shape=y.shape)


def data_iter(X, Y, batch_size):

    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    for i in range(0, n, batch_size):
        j = nd.array(indices[i:i+batch_size])
        yield X.take(j), Y.take(j)


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr/batch_size * param.grad

def l2loss(yhat, y):
    return (yhat-y.reshape(yhat.shape))**2/2

def train(epochs, lr, batch_size, net, X, Y, loss):

    for e in range(epochs):

        total_loss = 0
        start = time.time()
        for x, y in data_iter(X, Y, batch_size):
            with autograd.record():
                yhat = net(x)
                l = loss(yhat, y)
            l.backward()
            sgd(net.all_params(), lr, batch_size)
            total_loss += l.sum().asscalar()

        print('epoch: %d, loss: %f, time: %f sec' %(e+1, total_loss/len(Y), time.time()-start))

