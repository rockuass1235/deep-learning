import random
import cv2
from mxnet import nd, image
from mxnet.gluon import data as gdata, nn
import sys
import os
import _pickle as pk
import time



class SSD_Expand(nn.Block):

    def __init__(self, **kwargs):
        super(SSD_Expand, self).__init__(**kwargs)

    def forward(self, old_x, y):

        h, w, _ = old_x.shape

        img = nd.zeros(shape=old_x.shape, dtype='uint8')
        Y = nd.zeros(shape=y.shape, ctx=y.context)

        r = random.uniform(0.4, 0.8)
        size = int(w * r), int(h * r)

        aug = gdata.vision.transforms.Resize(size)

        dx = random.randint(0, w - size[0])
        dy = random.randint(0, h - size[1])

        img[dy:dy + size[1], dx:dx + size[1], :] = aug(old_x)

        for i in range(len(y)):
            if y[i, 0] >= 0:
                Y[i, 0] = y[i, 0]
                Y[i, 1:] = y[i, 1:] * nd.array([size[0], size[1], size[0], size[1]])
                Y[i, 1:] += nd.array([dx, dy, dx, dy])
                Y[i, 1:] /= nd.array([w, h, w, h])
            else:
                Y[i] = nd.ones(shape=(5)) * -1

        return img, Y


if __name__ == '__main__':

    argv = sys.argv[1:]
    data_dir = argv[0]
    if data_dir[-1] != '/':
        data_dir += '/'

    lst_file = argv[1]
    rec_file = argv[2]

    lst = None
    rec = None
    with open(lst_file, 'rb') as f:
        lst = pk.load(f)
    with open(rec_file, 'rb') as f:
        rec = pk.load(f)
    rec = nd.array(rec)

    if not os.path.exists(data_dir):
        raise RuntimeError('data path does not exsist')

    if not os.path.exists(data_dir+'augs/'):
        os.makedirs(data_dir+'augs')

    n = len(lst)
    X = []
    Y = []
    aug = SSD_Expand()



    start = time.time()
    for i in range(n):
        print('%d / %d' %(i+1, n))

        img = image.imread(data_dir+lst[i])
        label = rec[i]

        aug_img, aug_label = aug(img, label)

        aug_img = cv2.cvtColor(aug_img.asnumpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(data_dir+'augs/fix_'+lst[i], aug_img)
        X.append('augs/fix_'+lst[i])
        Y.append(aug_label.expand_dims(axis = 0))

    Y = nd.concat(*Y, dim = 0)
    with open(lst_file[:-4]+'_aug.lst', 'wb') as f:
        pk.dump(X, f)
    with open(rec_file[:-4]+'_aug.rec', 'wb') as f:
        pk.dump(Y.asnumpy(), f)

    print('success')
    print('total time: %f' %(time.time()-start))







