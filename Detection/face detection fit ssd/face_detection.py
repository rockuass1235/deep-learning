#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, data as gdata, nn
import time
import _pickle as pkl
import numpy as np
from SSD import *





X,Y = None, None
with open('face_data.pkl', 'rb') as f:
    X = pkl.load(f)
    Y = pkl.load(f)
    
augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(size = (256,256)),
                                        gdata.vision.transforms.ToTensor()])


print(Y[0])
dataset_train = gdata.ArrayDataset(X[:-50], Y[:-50])
dataset_test = gdata.ArrayDataset(X[-50:], Y[-50:])

    
        
        


# In[2]:


ctx, net = d2l.try_gpu(), MySSD(1)
print('training on', ctx)
net.initialize(init = init.Xavier(), ctx = ctx, force_reinit = True)
net.hybridize()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03,'momentum':0.9, 'wd': 5e-4})
net.load_parameters('face_detect.params')


# In[ ]:


import time


batch_size = 32
epochs = 50




net.load_parameters('face_detect.params')
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

            
            


# In[3]:


def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx))
    
    
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds.flatten(), anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    return output[0, idx]


def display(img, output, threshold):
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'r')




for x, y in zip(X[-50:], Y[-50:]):
    
    img = x
    x = augs(img).expand_dims(axis = 0)
    x = x.as_in_context(ctx)
    start = time.time()

    output = predict(x)
    d2l.set_figsize((5, 5))

    print('time: ', time.time()-start)
    display(img, output, threshold=0.8)
    d2l.plt.show()


# In[ ]:




