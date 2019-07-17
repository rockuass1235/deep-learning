from mxnet import nd, image
from mxnet.gluon import nn
import math

def G(x, y, o):
    return math.exp(-(x**2 + y**2)/(2 * o**2)) / (2 * math.pi * o**2)


class GuassianFilter(nn.Block):

    def __init__(self, size, o = 1.5,**kwargs):
        super(GuassianFilter, self).__init__(**kwargs)
        self.size = size
        self.o = o
        self.blk = nn.Conv2D(1, kernel_size=size, padding=(self.size-1)//2)

    def initialize(self, ctx=None, verbose=False, force_reinit=False):
        super(GuassianFilter, self).initialize()
        mid = (self.size-1)//2

        k = nd.zeros(shape=(1, 1, self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                k[:, :, i, j] = G(j-mid, mid-i, 1.5)
        k = k / k.sum().asscalar()
        self.blk.weight.set_data(k)

    def forward2d(self, x):
        shape = x.shape
        x = x.reshape((1, 1, *x.shape))
        x = x.astype('float32')
        x = self.blk(x)
        x.astype('uint8')
        return x.reshape((shape[0], shape[1]))

    def forward(self, x):

        if len(x.shape) < 3:
            return self.forward2d(x)
        else:
            yhat = nd.zeros(shape=(x.shape[2], x.shape[0], x.shape[1]))
            x = x.transpose((2,0,1))

            for y, x in zip(yhat, x):
                y[:] = self.forward2d(x)
            return yhat.transpose((1,2,0))














