
import mxnet
from mxnet import nd, autograd
from matplotlib import pyplot as plt
import function as fun
import model as md



dims = 2
x,y = fun.get_data(dims = dims, num = 1000)
w = nd.random.normal(scale=0.01, shape=(dims, 1))
b = nd.zeros(shape=(1,))

net = md.Net()
net.add(md.Layer(2, 1))


epochs = 50
lr = 0.03
batch_size = 16
loss = fun.l2loss

fun.train(epochs, lr, batch_size, net, x, y, loss)
print(net.all_params())




