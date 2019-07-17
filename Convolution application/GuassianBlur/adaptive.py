from mxnet import nd
from mxnet.gluon import nn

class AdaptiveThreshold(nn.Block):

    def __init__(self,size = 3,  **kwargs):
        super(AdaptiveThreshold, self).__init__(**kwargs)
        self.size = size
        self.blk = nn.Conv2D(1, size, padding=(self.size-1)//2)


    def initialize(self, ctx=None, verbose=False, force_reinit=False):
        super(AdaptiveThreshold, self).initialize()
        r = 1/self.size**2
        k = nd.ones(shape=(1,1,self.size, self.size)) * (-r)
        k[0,0,(self.size-1)//2,(self.size-1)//2] += 1
        self.blk.weight.set_data(k)

    def forward(self, x):
        shape = x.shape
        x = x.reshape((1,1,*x.shape))
        x = x.astype('float32')
        x = self.blk(x)
        x = x > 0
        x = x*255
        x.astype('uint8')
        return x.reshape((shape[0], shape[1]))









