from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata, loss as gloss
import time
import alexnet as alex

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



train_data = gdata.vision.CIFAR10(train=True)
test_data = gdata.vision.CIFAR10(train=False)


transformer = gdata.vision.transforms.ToTensor()
net = alex.TinyAlexNet(10)
net.initialize(init.Xavier())

epochs = 5*20
lr = 0.01
wd = 0.001
batch_size = 256
loss = gloss.SoftmaxCrossEntropyLoss()
mom = 0.9
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': mom})


net.load_parameters('TinyAlex.params')
train(epochs, batch_size, net, loss, trainer, train_data,transformer)
net.save_parameters('TinyAlex.params')



test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size = 256, shuffle = True)
total = 0
for x, y in test_iter:

    y = y.astype('float32')
    yhat = net(x)
    total += (yhat.argmax(axis=1) == y).sum().asscalar()
print('acc: ', total/len(test_data))






