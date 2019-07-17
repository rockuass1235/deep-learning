from mxnet import nd, image
from mxnet.gluon import nn
import matplotlib.pyplot as plt
import gray
import adaptive as adp


net = nn.Sequential()
net.add(gray.GrayFilter())
net.add(adp.AdaptiveThreshold(size= 63))

for layer in net:
    layer.initialize()

X = image.imread('sudoku-original.jpg')
yhat = net(X)
yhat = yhat.astype('uint8')



plt.imshow(yhat.asnumpy(), plt.cm.gray)
plt.show()