import os
import mxnet as mx
from mxnet import image, nd
from mxnet.gluon import data as gdata
import random
import numpy as np
import dlib
import ssd as mod


def fr_dataset(path):
    return FaceDataset(path)


def triplet_dataset(path, threshold=0.5):
    net = mod.MyMod(2)
    net.load_parameters('face_detect.params')
    net.reset_ctx(mx.gpu())
    augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(size=(256, 256)),
                                            gdata.vision.transforms.ToTensor()])

    ls = read_img(path)
    X = nd.zeros(shape=(1, 128, 128, 3))
    Y = []

    for i, p in enumerate(ls):

        img = image.imread(p)

        imgs = apply(img.expand_dims(axis=0), augs)
        bbox = net.predict(imgs, threshold=threshold)[0]
        if bbox.sum().asscalar() <= 0:
            continue
        label = int(p.split('/')[-2])
        Y.append(label)

        face = get_face(img, bbox).astype(X.dtype)

        X = nd.concat(X, face.expand_dims(axis=0), dim=0)

    return TripletDataset(X[1:], nd.array(Y).reshape(-1, 1))


def get_face_data(path, threshold = 0.5):
    net = mod.MyMod(2)
    net.load_parameters('face_detect.params')
    net.reset_ctx(mx.gpu())
    augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(size=(256, 256)),
                                            gdata.vision.transforms.ToTensor()])

    ls = read_img(path)
    X = nd.zeros(shape=(1, 128, 128, 3))
    Y = []

    for i, p in enumerate(ls):

        img = image.imread(p)

        imgs = apply(img.expand_dims(axis=0), augs)
        bbox = net.predict(imgs, threshold=threshold)[0]
        if bbox.sum().asscalar() <= 0:
            continue
        label = int(p.split('/')[-2])
        Y.append(label)

        face = get_face(img, bbox).astype(X.dtype)

        X = nd.concat(X, face.expand_dims(axis=0), dim=0)

    return gdata.ArrayDataset(X[1:], nd.array(Y).reshape(-1, 1))



def get_face(img, bbox):
    h, w, _ = img.shape
    aug = gdata.vision.transforms.Resize(size=(128, 128))
    det = nd.relu(bbox[2:])
    for a in det:
        if a > 1:
            a[:] = 1

    det = det * nd.array([w, h, w, h])
    x1, y1, x2, y2 = det.asnumpy().astype(int)

    face = img[y1:y2, x1:x2]
    face = fix(face)

    return aug(face)


def apply(X, augs):
    img = nd.zeros((256, 256, 3))
    img = augs(img)
    shape = (len(X),) + img.shape
    Xhat = nd.zeros(shape)

    for i, x in enumerate(X):
        Xhat[i] = augs(x)
    return Xhat


def fix(img):
    h, w, _ = img.shape
    n = max(h, w)
    x = nd.zeros(shape=(n, n, 3))
    x[:h, :w, :] = img

    return x


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


class FaceDataset(gdata.Dataset):

    def __init__(self, path, get_data_first=False):
        self.ls = read_img(path)
        self._first = get_data_first

        self.Y = nd.zeros(shape=(len(self), 10, 5))

        for i in range(self.__len__()):
            y = self.__get_label(i)
            self.Y[i] = y

    def __len__(self):
        return len(self.ls)

    def __get_label(self, idx):

        detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        X = image.imread(self.ls[idx])
        X = fix(X).astype('uint8')
        h, w, _ = X.shape

        # 資料的ground truth數量不一同無法使用batch size trainning，
        # 可補[0,0,0,0,0] 將其大小擴展為一致，MultiboxTarget計算時會自動忽略不影響輸出
        dets = detector(X.asnumpy(), 1)
        Y = nd.zeros(shape=(10, 5))
        for j, det in enumerate(dets):
            x1, y1, x2, y2 = det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()
            Y[j] = nd.array([0, x1 / w, y1 / h, x2 / w, y2 / h])
        return Y

    def __getitem__(self, idx):

        img = image.imread(self.ls[idx])
        img = fix(img)
        return img, self.Y[idx]


class _TransformAllClosure(object):
    """Use callable object instead of nested function, it can be pickled."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args):
        return tuple(self._fn(x) for x in args)


class TripletDataset(gdata.Dataset):

    def __init__(self, X, Y):

        if not isinstance(X, nd.NDArray):
            raise Exception('type of X is not nd.NDArray')
        if not isinstance(Y, nd.NDArray):
            raise Exception('type of Y is not nd.NDArray')

        self.X = X
        self.Y = Y
        self.classes = np.unique(self.Y.asnumpy())
        self.cls_num = len(self.classes)
        self.groups = tuple(
            np.where(Y.asnumpy() == cls)[0] for cls in self.classes)  # np.where return 座標(tuple)故取idx[0]
        self.pairs = self._get_pairs()

    def __getitem__(self, idx):

        idx = self.pairs[idx]

        return self.X[idx][0], self.X[idx][1], self.X[idx][2]

    def _get_pairs(self):

        pairs = []
        for i, indeces in enumerate(self.groups):

            np.random.shuffle(indeces)

            for j in range(len(indeces) - 1):
                piv = indeces[j]
                pos = indeces[j + 1]

                neg = i + random.randint(1, self.cls_num - 1)
                neg %= self.cls_num
                neg_idx = random.randint(0, len(self.groups[neg]) - 1)  # a <= random <= b
                neg = self.groups[neg][neg_idx]

                pairs.append(nd.array([piv, pos, neg]))

        return pairs

    def __len__(self):

        return len(self.pairs)

    def transform_all(self, fn, lazy=True):

        return self.transform(_TransformAllClosure(fn), lazy)




