import mxnet as mx
from mxnet import gluon, init, contrib, nd
from mxnet.gluon import nn, model_zoo
import numpy as np


def cls_blk(anchors_num, classes):
    out = anchors_num * (classes)
    net = nn.Conv2D(out, kernel_size=3, padding=1)
    net.initialize(init.Xavier())
    return net


def bbox_blk(anchors_num):
    out = anchors_num * 4
    net = nn.Conv2D(out, kernel_size=3, padding=1)
    net.initialize(init.Xavier())
    return net


class MyBlk(nn.Block):

    def __init__(self, blk, classes, size, ratio, **kwargs):
        super(MyBlk, self).__init__(**kwargs)

        self.size = size
        self.ratio = ratio
        N = len(size) + len(ratio) - 1

        self.blk = blk
        self.cls_blk = cls_blk(N, classes)
        self.bbox_blk = bbox_blk(N)

    def forward(self, x):
        yhat = self.blk(x)
        anchors = contrib.nd.MultiBoxPrior(yhat, sizes=self.size, ratios=self.ratio)
        cls_yhat = self.cls_blk(yhat)
        bbox_yhat = self.bbox_blk(yhat)

        return yhat, anchors, cls_yhat, bbox_yhat


class MyMod(nn.Block):

    def __init__(self, classes, ctx = mx.cpu(), **kwargs):
        super(MyMod, self).__init__(**kwargs)
        self.classes = classes
        self.ctx = ctx

        net = gluon.model_zoo.vision.mobilenet_v2_1_0(pretrained=True).features

        self.net_0 = MyBlk(net[:4], classes, [0.8/(2**8.5), 0.8/(2**8)], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_1 = MyBlk(net[4:6], classes, [0.8/(2**7), 0.8/(2**6.5)], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_2 = MyBlk(net[6:9], classes, [0.8/(2**5.5), 0.8/(2**5)], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_3 = MyBlk(net[9:16], classes, [0.8/(2**4.5), 0.8/(2**4), 0.8/(2**3.5)], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_4 = MyBlk(net[16:23], classes, [0.8/(2**3), 0.8/(2**2.5), 0.8/(2**2)], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_5 = MyBlk(net[23], classes, [0.8/(2**1.5), 0.8/(2**0.5), 0.8/(2**0)], [0.5, 1, 2, 1.618, 1 / 1.618])

    def forward(self, x):
        anchors, cls_yhats, bbox_yhats = [], [], []

        for i in range(6):
            net = getattr(self, 'net_%d' % i)
            x, anch, cls_yhat, bbox_yhat = net(x)

            cls_yhat = cls_yhat.transpose((0, 2, 3, 1)).flatten()
            cls_yhat = cls_yhat.reshape((cls_yhat.shape[0], -1, self.classes))

            bbox_yhat = bbox_yhat.transpose((0, 2, 3, 1)).flatten()
            bbox_yhat = bbox_yhat.reshape((bbox_yhat.shape[0], -1, 4))

            anchors.append(anch)
            cls_yhats.append(cls_yhat)
            bbox_yhats.append(bbox_yhat)

        return nd.concat(*anchors, dim=1), nd.concat(*cls_yhats, dim=1), nd.concat(*bbox_yhats, dim=1)
