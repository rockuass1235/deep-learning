from mxnet import nd, image
from mxnet.gluon import nn
import matplotlib.pyplot as plt
import gray


net = nn.Sequential()
net.add(gray.GrayFilter())


for layer in net:
    layer.initialize()

X = image.imread('lena.jpg')
yhat = net(X)
yhat = yhat.astype('uint8')



plt.imshow(yhat.asnumpy(), plt.cm.gray)
plt.show()