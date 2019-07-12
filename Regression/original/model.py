import mxnet as mx
from mxnet import nd, autograd


class Layer(object):

    def __init__(self, intput, output):
        self.w = nd.random.normal(scale=0.01, shape=(intput, output))
        self.b = nd.zeros(shape=(output,))
        self.w.attach_grad()
        self.b.attach_grad()

    def __call__(self, x):
        return nd.dot(x,self.w) + self.b

class Activation(object):
    def __init__(self, act):
        if act == 'relu':
            self.act = nd.relu
        elif act == 'sigmoid':
            self.act = nd.sigmoid
        elif act == 'tanh':
            self.act = nd.tanh

    def __call__(self, x):
        return self.act(x)



class Net(object):

    def __init__(self):
        self.__layer = []
    def add(self, layer):

        self.__layer.append(layer)
    def all_params(self):

        params = []
        for l in self.__layer:
            params.append(l.w)
            params.append(l.b)
        return params
    def __call__(self, x):

        for  l in self.__layer:
            x = l(x)
        return x




