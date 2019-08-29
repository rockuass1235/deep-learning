
import MyDataset as mydata
import mxnet as mx
from mxnet import autograd, contrib, gluon
from mxnet.gluon import loss as gloss, data as gdata
import time
import ssd as mod

start = 0

# ==========================================================================================================
#                               data auguments prepare
# ==========================================================================================================


start = time.time()
print('data initializing.............')

path = 'lfw_5590'

augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(size=(256, 256)),
                                        gdata.vision.transforms.ToTensor()])
dataset = mydata.fr_dataset(path)

print('data initializing success, time: %.3f' % (time.time() - start))

# ==========================================================================================================
#                               Model
# ==========================================================================================================


start = time.time()
print('model initializing.............')

ctx, net = mx.gpu(), mod.MyMod(2)
net.reset_ctx(ctx)
print('training on', ctx)

print('model initializing success, time: %.3f' % (time.time() - start))

# ==========================================================================================================
#                               environment
# ==========================================================================================================

start = time.time()
print('environment initializing.............')

net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03, 'momentum': 0.9, 'wd': 5e-4})

batch_size = 32
epochs = 5


def cost(cls_yhat, cls_y, bbox_yhat, bbox_y, bbox_masks):
    cls_cost = gloss.SoftmaxCrossEntropyLoss()
    bbox_cost = gloss.L1Loss()

    c_l = cls_cost(cls_yhat, cls_y)
    b_l = bbox_cost(bbox_yhat.flatten() * bbox_masks, bbox_y * bbox_masks)
    return c_l + b_l


print('environment initializing success, time: %.3f' % (time.time() - start))
# ==========================================================================================================
#                               training part
# ==========================================================================================================


print('start training.............')

# net.load_parameters('face_detect.params')
for e in range(epochs):

    train_iter = gdata.DataLoader(dataset.transform_first(augs), batch_size=batch_size, shuffle=True)
    total_loss = 0
    start = time.time()
    n = 0

    for x, y in train_iter:
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)

        with autograd.record():
            anchors, cls_yhat, bbox_yhat = net(x)
            bbox_y, bbox_masks, cls_y = contrib.nd.MultiBoxTarget(anchors, y, cls_yhat.transpose((0, 2, 1)))

            l = cost(cls_yhat, cls_y, bbox_yhat, bbox_y, bbox_masks)

        l.backward()
        trainer.step(batch_size)
        total_loss += l.sum().asscalar()
        n += len(y)

    print('epoch: %02d, train loss: %.6f, time: %.3f sec' % (e + 1, total_loss / n, time.time() - start))
    # net.save_parameters('face_detect.params')
