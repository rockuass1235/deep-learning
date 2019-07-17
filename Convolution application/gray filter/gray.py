from mxnet import nd
from mxnet.gluon import nn



class GrayFilter(nn.Block):

    def __init__(self, **kwargs):
        super(GrayFilter, self).__init__(**kwargs)
        self.blk = nn.Conv2D(1, 1)

    def initialize(self, ctx=None, verbose=False,force_reinit=False):
        super(GrayFilter, self).initialize()
        k = [[[[299]], [[587]], [[114]]]]
        k = nd.array(k)
        b = nd.ones(shape=(1,))*500
        self.blk.weight.set_data(k)
        self.blk.bias.set_data(b)


    def forward(self, x):
        shape = x.shape
        x = x.reshape((1, *x.shape))
        x = x.transpose((0, 3, 1, 2))
        x = x.astype('float32')
        x = self.blk(x)
        x = x/1000
        x.astype('uint8')
        return x.reshape((shape[0], shape[1]))





