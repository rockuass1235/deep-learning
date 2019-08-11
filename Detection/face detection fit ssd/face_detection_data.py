#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
from mxnet import image, nd
import numpy as np
import dlib
import matplotlib.pyplot as plt
import d2lzh as d2l




def show_bbox(img, bboxes):
    
    fig = d2l.plt.imshow(img.asnumpy())
    bboxes = bboxes[:, 1:]
    for bbox in bboxes:
        if bbox.sum() == 0:
            continue
        else:
            rect = d2l.bbox_to_rect(bbox.asnumpy(), 'r')
            fig.axes.add_patch(rect)
            
    plt.show()


    
    

def read_img(path):
    
    if path[-1] != '/':
        path += '/'
    
    li = []
    if os.path.isdir(path):
        files = os.listdir(path)
        
        for file in files:
            li += read_img(path+file)
            
    else:
        li.append(path[:-1])
    return li
        

def get_data(path):
    
    dir = read_img(path)
    X = nd.zeros(shape = (len(dir), 250, 250, 3), dtype = 'uint8')
    for x, d in zip(X, dir):
        img = image.imread(d)
        x[:] = img
        
        
         
    # 資料的ground truth數量不一同無法使用batch size trainning，
    # 可補[0,0,0,0,0] 將其大小擴展為一致，MultiboxTarget計算時會自動忽略不影響輸出
    Y = nd.zeros(shape = (len(dir), 10, 5))
    
    
    
    
    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    
    for i, x in enumerate(X):
        dets = detector(x.asnumpy(), 1)
        gt = nd.zeros(shape = (10, 5))
       
        for j, det in enumerate(dets):
            x1, y1, x2, y2 = det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()
            gt[j] = nd.array([0, x1, y1, x2, y2])
        
        Y[i] = gt/250
            
        
    return X, Y

    
    
    


# In[2]:


import time


start = time.time()
path = 'lfw_5590'
X,Y = get_data(path)
print('time: ', time.time()-start)


# In[3]:


print(X.shape)
print(Y.shape)


# In[4]:


import _pickle as pkl


with open('face_data.pkl', 'wb') as f:
    
    pkl.dump(X, f)
    pkl.dump(Y, f)


# In[ ]:




