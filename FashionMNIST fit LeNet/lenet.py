from mxnet.gluon import nn

def LeNet():

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(6, kernel_size=5, strides=1, activation='tanh'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Conv2D(16, kernel_size=5, strides=1, activation='tanh'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Flatten())
        net.add(nn.Dense(120, activation='tanh'))
        net.add(nn.Dense(84, activation='tanh'))
        net.add(nn.Dense(10))

    return net
print(LeNet())