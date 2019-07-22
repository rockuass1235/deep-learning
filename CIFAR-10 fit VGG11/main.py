import mxnet as mx
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata, loss as gloss
import time
import vgg

def train(epochs, batch_size, net, loss, trainer, train_data, transformer, ctx):
    train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size=batch_size, shuffle=True)
   
    
    for e in range(epochs):

        total_loss = 0
        start = time.time()
        for x, y in train_iter:
            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx).astype('float32')
                
            
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
net = vgg.VGG(10, 4)
net.initialize(init.Xavier())

epochs = 200
lr = 0.01
wd = 0.001
batch_size = 256
loss = gloss.SoftmaxCrossEntropyLoss()
mom = 0.9


ctx = mx.gpu()
net.collect_params().reset_ctx(ctx)
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': mom, 'wd': wd})


#net.load_parameters('TinyVGG11.params')
train(epochs, batch_size, net, loss, trainer, train_data,transformer, ctx)
net.save_parameters('TinyVGG11.params')



test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size = 256, shuffle = True)
total = 0
net.collect_params().reset_ctx(mx.cpu())
for x, y in test_iter:

    y = y.astype('float32')
    yhat = net(x)
    total += (yhat.argmax(axis=1) == y).sum().asscalar()
print('acc: ', total/len(test_data))

