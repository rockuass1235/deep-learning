#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, data as gdata, nn
import time
import _pickle as pkl
import numpy as np
from SSD import *
import time










#==========================================================================================================
#                               data prepare
#==========================================================================================================



X,Y = None, None
with open('face_data.pkl', 'rb') as f:
    X = pkl.load(f)
    Y = pkl.load(f)
    
augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(size = (256,256)),
                                        gdata.vision.transforms.ToTensor()])


print(Y[0])
dataset_train = gdata.ArrayDataset(X[:-50], Y[:-50])
dataset_test = gdata.ArrayDataset(X[-50:], Y[-50:])











#==========================================================================================================
#                               Model
#==========================================================================================================




def predict(net, X, ctx):
    
    X = X.as_in_context(ctx)
    net.collect_params().reset_ctx(ctx)
    
    
    
    
    anchors, cls_yhat, bbox_yhat = net(X)
    cls_yhat = cls_yhat.softmax().transpose((0, 2, 1))
    out = contrib.nd.MultiBoxDetection(cls_yhat, bbox_yhat.flatten(), anchors)
    bboxes = []
    
    for i, img in enumerate(out):
        
        idx = img[:, 0] > -1
        idx = np.where(idx.asnumpy() >= 0.5)
        bboxes.append(img[idx].as_in_context(mx.cpu()))
    
    return bboxes




ctx, net = d2l.try_gpu(), MySSD(1)
print('training on', ctx)




#==========================================================================================================
#                               environment
#==========================================================================================================






net.initialize(init = init.Xavier(), ctx = ctx, force_reinit = True)
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03,'momentum':0.9, 'wd': 5e-4})




batch_size = 32
epochs = 50

#==========================================================================================================
#                               training part
#==========================================================================================================



#net.load_parameters('face_detect.params')
for e in range(epochs):
    
    train_iter = gdata.DataLoader(dataset_train.transform_first(augs), batch_size = batch_size, shuffle=True)
    total_loss = 0
    start = time.time()
    n = 0
    
    for x, y in train_iter:
       
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        
        with autograd.record():
            
            anchors, cls_yhat, bbox_yhat = net(x)
            bbox_y, bbox_masks, cls_y = get_label(anchors, y, cls_yhat)
            l = cost(cls_yhat, cls_y, bbox_yhat, bbox_y, bbox_masks)
        
        l.backward()
        trainer.step(batch_size)
        total_loss += l.sum().asscalar()
        n += len(y)
    
    print('epoch: %02d, train loss: %.6f, time: %.3f sec' %(e+1, total_loss/n, time.time()-start))
    net.save_parameters('face_detect.params')
    
    
    



