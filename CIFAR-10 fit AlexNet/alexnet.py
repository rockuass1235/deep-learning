from mxnet.gluon import nn

def AlexNet(out):

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=3, strides=2))
        net.add(nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=3, strides=2))
        net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=3, strides=2))
        net.add(nn.Dense(4096, activation="relu"))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(4096, activation="relu"))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(out))

    return net


def TinyAlexNet(out):
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(32, kernel_size=5, padding=2, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Conv2D(32, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.Conv2D(32, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(out))
    return net

