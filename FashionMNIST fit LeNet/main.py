from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata, loss as gloss
import time
import lenet as le


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def train(epochs, batch_size, net, loss, trainer, train_data, transformer):
    train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size=batch_size, shuffle=True)
    for e in range(epochs):

        total_loss = 0
        start = time.time()
        for x, y in train_iter:
            y = y.astype('float32')
            with autograd.record():
                yhat = net(x)
                l = loss(yhat, y)
            l.backward()
            trainer.step(batch_size)
            total_loss += l.sum().asscalar()

        print('epoch: %d, loss: %f, time: %f sec' % (e + 1, total_loss / len(train_data), time.time() - start))



train_data = gdata.vision.FashionMNIST(train=True)
test_data = gdata.vision.FashionMNIST(train=False)

transformer = gdata.vision.transforms.ToTensor()
net = le.LeNet(10)
net.initialize(init.Xavier())

epochs = 5*20
lr = 0.03
wd = 0.001
batch_size = 64
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'wd': wd})


net.load_parameters('lenet.params')
train(epochs, batch_size, net, loss, trainer, train_data,transformer)
net.save_parameters('lenet.params')



test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size = 256, shuffle = True)
total = 0
for x, y in test_iter:

    y = y.astype('float32')
    yhat = net(x)
    print(yhat.argmax(axis=1) == y)
    total += (yhat.argmax(axis=1) == y).sum().asscalar()
print('acc: ', total/len(test_data))






