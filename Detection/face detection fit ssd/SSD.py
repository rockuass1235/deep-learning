#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mxnet as mx
from mxnet import gluon, nd, contrib
from mxnet.gluon import nn, model_zoo, loss as gloss




def blk(num, channels):

    net = nn.Sequential()
    with net.name_scope():
        for _ in range(num):
            net.add(nn.Conv2D(channels, kernel_size=3, padding=1))
            net.add(nn.BatchNorm(in_channels=channels))
            net.add(nn.Activation('relu'))
            
            
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
    return net



def cls_blk(anchors_num, cls_num):
    
    out = anchors_num * (cls_num+1)
    
    return nn.Conv2D(out, kernel_size=3, padding = 1)

def bbox_blk(anchors_num):
    
    out = anchors_num * 4
    
    return nn.Conv2D(out, kernel_size=3, padding = 1)





class MyBlk(nn.Block):
    
    def __init__(self, blk, cls_num, size, ratio, **kwargs):
        super(MyBlk, self).__init__(**kwargs)
        
        
        self.size = size
        self.ratio = ratio
        N = len(size) + len(ratio) - 1
        
        self.blk = blk
        self.cls_blk = cls_blk(N, cls_num)
        self.bbox_blk = bbox_blk(N)
        self.cls_blk.initialize(init = init.Xavier())
        self.bbox_blk.initialize(init = init.Xavier())
        
        
        
        
    def forward(self, x):
        
        
        yhat = self.blk(x)
        anchors = contrib.nd.MultiBoxPrior(yhat, sizes = self.size, ratios = self.ratio)
        cls_yhat = self.cls_blk(yhat)
        bbox_yhat = self.bbox_blk(yhat)
        
        return yhat, anchors, cls_yhat, bbox_yhat

    

class MySSD(nn.Block):
    
    
    def __init__(self, cls_num, **kwargs):
        
        super(MySSD, self).__init__(**kwargs)
        self.num = cls_num
        
        net = nn.Sequential()
        with net.name_scope():
            net.add(blk(1, 16))
            net.add(blk(1, 32))
            net.add(blk(2, 64))
                 
        self.net_0 = MyBlk(net, cls_num, [0.2, 0.272], [1, 2, 0.5])
        self.net_1 = MyBlk(blk(2, 128), cls_num, [0.37, 0.447], [1, 2, 0.5])
        self.net_2 = MyBlk(blk(2, 128), cls_num, [0.54, 0.619], [1, 2, 0.5])
        self.net_3 = MyBlk(nn.GlobalMaxPool2D(), cls_num, [0.71, 0.79], [1, 2, 0.5])
        
        self.net_3.initialize(init = init.Xavier())
        
        
               
    def forward(self, x):
        
        anchors, cls_yhats, bbox_yhats = [], [], []
        for i in range(4):
            net = getattr(self, 'net_%d' %i)
            x, anch, cls_yhat, bbox_yhat = net(x)
            
            
            cls_yhat = cls_yhat.transpose((0,2,3,1)).flatten()
            cls_yhat = cls_yhat.reshape((cls_yhat.shape[0], -1, self.num+1))
            
            
            bbox_yhat = bbox_yhat.transpose((0,2,3,1)).flatten()
            bbox_yhat = bbox_yhat.reshape((bbox_yhat.shape[0], -1, 4))
            
            
            anchors.append(anch)
            cls_yhats.append(cls_yhat)
            bbox_yhats.append(bbox_yhat)
        
        return nd.concat(*anchors, dim = 1), nd.concat(*cls_yhats, dim = 1), nd.concat(*bbox_yhats, dim = 1)
    
    
def get_label(anchors, y, yhat):
    
    bbox_y, bbox_masks, cls_y = contrib.nd.MultiBoxTarget(anchors, y, yhat.transpose((0, 2, 1)))
    return bbox_y, bbox_masks, cls_y



cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()

def cost(cls_yhat, cls_y, bbox_yhat, bbox_y, bbox_masks):
    
    bbox_yhat = bbox_yhat.flatten()
    return cls_loss(cls_yhat, cls_y) + bbox_loss(bbox_yhat*bbox_masks, bbox_y * bbox_masks)


def predict(net, x):
    
    
    anchors, cls_yhat, bbox_yhat = net(x.as_in_context(ctx))
    
    cls_yhat = cls_yhat.softmax()
    cls_yhat = cls_yhat.transpose((0, 2, 1))  # shape = batch_size, cls_one_hot, num_anchors
    bbox_yhat = bbox_yhat.flatten()
    
    return contrib.nd.MultiBoxDetection(cls_yhat, bbox_yhat, anchors, nms_threshold= 0.5)
            
            
        

