import os
import mxnet as mx
from mxnet import nd, image
import d2lzh as d2l
import ssd as mod
import matplotlib.pyplot as plt
from mxnet.gluon import data as gdata
import MyDataset as mydata

ctx = mx.gpu()


def apply(X, augs):
    img = nd.zeros((256, 256, 3))
    img = augs(img)
    shape = (len(X),) + img.shape
    Xhat = nd.zeros(shape)

    for i, x in enumerate(X):
        Xhat[i] = augs(x)
    return Xhat


def show_img(imgs):
    cols = len(imgs)
    _, axes = plt.subplots(1, cols)

    for i in range(cols):
        axes[i].imshow(imgs[i])
        axes[i].axes.get_xaxis().set_visible(False)
        axes[i].axes.get_yaxis().set_visible(False)

    return axes


def get_face(img, threshold=0.5):
    net = mod.MyMod(2)
    net.load_parameters('face_detect.params')
    net.reset_ctx(ctx)
    augs = gdata.vision.transforms.Compose(
        [gdata.vision.transforms.Resize(size=(256, 256)), gdata.vision.transforms.ToTensor()])

    img = fix(img)
    imgs = apply(img.expand_dims(axis=0), augs)
    bbox = net.predict(imgs, threshold=threshold)[0]

    h, w = img.shape[0:2]
    img = img.astype('uint8')

    if bbox.sum().asscalar() <= 0:
        return img, nd.zeros(shape=img.shape, dtype='uint8')

    det = nd.relu(bbox[2:])
    for a in det:
        if a > 1:
            a[:] = 1

    det = det * nd.array([w, h, w, h])
    x1, y1, x2, y2 = det.asnumpy().astype(int)

    return img, img[y1:y2, x1:x2]


def get_faces(path, threshold=0.5):
    ls = read_img(path)
    X = nd.zeros(shape=(len(ls), 128, 128, 3))

    for i, p in enumerate(ls):
        img = image.imread(p)
        img, face = get_face(img, threshold)
        X[i] = gdata.vision.transforms.Resize((128, 128))(face)

    return X


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


# ============================== Main ===================================================


path = 'face'
threshold = 0.7
f = get_faces(path, threshold)
print(f.shape)


