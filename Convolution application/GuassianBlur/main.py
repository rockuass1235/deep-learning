from mxnet import nd, image
from mxnet.gluon import nn
import matplotlib.pyplot as plt
import gray
import adaptive as adp
import gaussian as ga

net = nn.Sequential()
net.add(gray.GrayFilter())
net.add(ga.GuassianFilter(7, 1))
net.add(adp.AdaptiveThreshold(size=13))

for layer in net:
    layer.initialize()

X = image.imread('lena.jpg')
yhat = net(X)
yhat = yhat.astype('uint8')


plt.imshow(yhat.asnumpy(), plt.cm.gray)
plt.show()
