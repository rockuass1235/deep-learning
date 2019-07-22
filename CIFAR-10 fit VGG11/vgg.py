from mxnet.gluon import nn


def blk(channel, num):

    net = nn.Sequential()
    with net.name_scope():
        for _ in range(num):
            net.add(nn.Conv2D(channel, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
    return net




# 因為這個網絡使用了8個卷積層和3個全連接層，所以經常被稱為VGG-11。

def VGG(out,k = 1, conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]):

    net = nn.Sequential()
    with net.name_scope():
        for num, channel in conv_arch:
            net.add(blk(channel//k, num))


        net.add(nn.Flatten())
        net.add(nn.Dense(4096, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(4096, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(out))
    return net




