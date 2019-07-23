
# 前言

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/model_history.png)

在2012年這一年中，AlexNet為一個重大的突破，也開始了大CNN時代，AlexNet在LeNet的基礎上增加了3個卷積層。但AlexNet作者對它們的捲積窗口、輸出通道數和構造順序均做了大量的調整。

雖然AlexNet指明了深度卷積神經網絡可以取得出色的結果，但並沒有提供簡單的規則以指導後來的研究者如何設計新的網絡。

VGG，它的名字來源於論文作者所在的實驗室Visual Geometry Group [1]。VGG提出了可以通過重複使用簡單的基礎塊來構建深度模型的思路。




# VGG

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/vgg_block.png)

VGG塊的組成規律是：連續使用數個相同的填充為1、窗口形狀為 3×3 的捲積層後接上一個步幅為2、窗口形狀為 2×2 的最大池化層。卷積層保持輸入的高和寬不變，而池化層則對其減半。

它也可以指定卷積層的數量num_convs和輸出通道數num_channels。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/vgg.png)


VGG網絡由卷積層模塊後接全連接層模塊構成。卷積層模塊串聯數個vgg_block，其超參數由變量conv_arch定義。該變量指定了每個VGG塊裡卷積層個數和輸出通道數。全連接模塊則跟AlexNet中的一樣。


我們構造一個VGG網絡。它有5個卷積塊，前2塊使用單卷積層，而後3塊使用雙卷積層。第一塊的輸出通道是64，之後每次對輸出通道數翻倍，直到變為512。因為這個網絡使用了8​​個卷積層和3個全連接層，所以經常被稱為VGG-11。

```Python

def blk(channel, num):

    net = nn.Sequential()
    with net.name_scope():
        for _ in range(num):
            net.add(nn.Conv2D(channel, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
    return net


def VGG(out,conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]):

    net = nn.Sequential()
    with net.name_scope():
        for num, channel in conv_arch:
            net.add(blk(channel, num))


        net.add(nn.Flatten())
        net.add(nn.Dense(4096, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(4096, activation='relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(out))
    return net


```

# 使用方法

由於VGG的權重比起AlexNet來說簡直是爆炸性成長，訓練時間也是相當漫長，所以我改用gpu(GTX1070)進行運算

### cpu運算:

epoch: 1, loss: 2.301002, time: 1709.729176 sec

### gpu運算

epoch: 1, loss: 1.943757, time: 13.645320 sec



```Python

def train(epochs, batch_size, net, loss, trainer, train_data, transformer, ctx):
    train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size=batch_size, shuffle=True)
   
    
    for e in range(epochs):

        total_loss = 0
        start = time.time()
        for x, y in train_iter:
		
			# 將資料copy to gpu
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
net = VGG(10)
net.initialize(init.Xavier())

epochs = 100
lr = 0.01
wd = 0.001
batch_size = 256
loss = gloss.SoftmaxCrossEntropyLoss()
mom = 0.9



# ctx的改變需要在初始化trainer之前 否則訓練時trainer得到的權重位置依然在cpu上!!!!

ctx = mx.gpu()
net.collect_params().reset_ctx(ctx)
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})


net.load_parameters('TinyVGG11.params')
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


```


# 結果


epochs = 200
lr = 0.01
wd = 0.001
batch_size = 256
loss = gloss.SoftmaxCrossEntropyLoss()
mom = 0.9


acc:  0.7405












# 原文出處

http://zh.gluon.ai/chapter_convolutional-neural-networks/vgg.html

https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-cnn%E6%BC%94%E5%8C%96%E5%8F%B2-alexnet-vgg-inception-resnet-keras-coding-668f74879306