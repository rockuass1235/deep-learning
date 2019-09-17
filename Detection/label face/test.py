import matplotlib.pyplot as plt
from mxnet import image
import numpy as np
import _pickle as pk
import sys


def show_bbox(axes, img_shape, gts, color):
    h, w, _ = img_shape
    w -= 1
    h -= 1

    for gt in gts:

        if gt.shape[0] > 5:
            x1, y1, x2, y2 = (gt[2:] * np.array([w, h, w, h])).astype(int)
        else:
            x1, y1, x2, y2 = (gt[1:] * np.array([w, h, w, h])).astype(int)

        rect = plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fill=False, edgecolor=color, linewidth=2)
        axes.add_patch(rect)


if __name__ == '__main__':


    data_path = sys.argv[1]
    lst_path = sys.argv[2]
    rec_path = sys.argv[3]

    lst = None
    rec = None

    with open(lst_path, 'rb') as f:
        lst = pk.load(f)  # data type: list
    with open(rec_path, 'rb') as f:
        rec = pk.load(f)  # data type: numpy

    for i in range(len(lst)):
        img = image.imread(data_path + lst[i]).asnumpy()
        fig = plt.imshow(img)
        show_bbox(fig.axes, img.shape, rec[i], 'blue')
        plt.show()

