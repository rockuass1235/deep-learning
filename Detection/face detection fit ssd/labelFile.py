
#*********************************************************************
#                       get all dir
#*********************************************************************
import os

def read_img(path):
    if path[-1] != '/':
        path += '/'

    li = []
    if os.path.isdir(path):
        files = os.listdir(path)

        for file in files:
            li += read_img(path + file)

    else:
        li.append(path[:-1])
    return li


path = 'data/data/'
lst = read_img(path)
lst = [path[10:] for path in lst]

#*********************************************************************
#                   label each img
#*********************************************************************
import matplotlib.pyplot as plt
from mxnet import image, nd
import dlib
import time

MAX_gt = 20  # max ground truth shape


def show_bbox(axes, img, gts):
    h, w, _ = img.shape
    for gt in gts:
        if gt[0] < 0:
            continue

        x1, y1, x2, y2 = (gt[1:] * np.array([w * 0.9, h * 0.9, w * 1.1, h * 1.1])).astype(int)

        rect = plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fill=False, edgecolor='red', linewidth=2)
        axes.add_patch(rect)


def get_ground_truth(img):
    h, w, _ = img.shape
    detector = dlib.cnn_face_detection_model_v1('gluon_train/mmod_human_face_detector.dat')
    gts = nd.ones(shape=(MAX_gt, 5)) * -1
    dets = detector(img, 1)

    if len(dets) <= 0:
        return (False, gts)

    for i, det in enumerate(dets):
        if i >= 20:
            break
        x1, y1, x2, y2 = det.rect.left() / w, det.rect.top() / h, det.rect.right() / w, det.rect.bottom() / h
        gts[i] = nd.array([0, x1, y1, x2, y2])

    return (True, gts)


X = []
Y = nd.zeros(shape=(1, MAX_gt, 5))
n = len(lst)

start = time.time()
for i in range(n):

    img = image.imread(path + lst[i]).asnumpy()

    flg, gts = get_ground_truth(img)

    if flg:
        # fig = plt.imshow(img)
        # show_bbox(fig.axes, img, gts.asnumpy())
        # plt.show()
        X.append(lst[i])
        Y = nd.concat(Y, gts.expand_dims(axis=0), dim=0)

    print('%d / %d' % (i + 1, n))

print('time: %f' % (time.time() - start))



#*********************************************************************
#                   save as file
#*********************************************************************

import _pickle as pk

file_name = 'guilty_face'

with open(file_name+'.lst', 'wb') as f:
    pk.dump(X, f)  # data type: list
with open(file_name+'.rec', 'wb') as f:
    pk.dump(Y[1:].asnumpy(), f)    # data type: numpy

'''
#*********************************************************************
#                   test file
#*********************************************************************
from mxnet import image
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pk


def show_bbox(axes, img, gts):
    h, w, _ = img.shape
    for gt in gts:
        if gt[0] < 0:
            continue

        x1, y1, x2, y2 = (gt[1:] * np.array([w * 0.9, h * 0.9, w * 1.1, h * 1.1])).astype(int)

        rect = plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fill=False, edgecolor='red', linewidth=2)
        axes.add_patch(rect)


path = 'D:/data/'
file_name = 'guilty_face'
arr = None
labels = None

with open(file_name+'.lst', 'rb') as f:
    arr = pk.load(f)
with open(file_name+'.rec', 'rb') as f:
    labels = pk.load(f)



for p, gts in zip(arr, labels):
    img = image.imread(path + p).asnumpy()
    fig = plt.imshow(img)
    show_bbox(fig.axes, img, gts)
    plt.show()
'''
