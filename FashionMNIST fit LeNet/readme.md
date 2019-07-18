# LeNet

LeNet 由Yann LeCun於1998年提出，CNN核心架構即源自這篇paper，主要是用於手寫字體的識別，同時也是目前CNN的hello world。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/LeNet.png)

其架構可以歸納為 convolution layer > pooling layer > fully connective layer 


* 輸入 32x32 的圖像，經過第一層Convolution layer 的 6個 5x5 的卷積核產生 6 張 28x28 的特徵圖(整個輸入與一個卷積核產生1張圖)

* 在透過Max pooling layer 將 28x28大小的圖像的區域像素點處理變成 14x14大小的特徵圖

* 再將6張 14x14 特徵圖輸入第二層Convolution layer 的 16個 5x5 的卷積核產生 16 張 10x10 的特徵圖

* 透過Max pooling layer 將圖像的區域像素點處理變成 5x5大小的特徵圖

* 將所有特徵圖資訊變成 16x5x5 大小的一維陣列 輸入 Dense layer(120) -> Dense layer(84) -> Dense layer(10)

這是一個很典型的LeNet-5 (5維卷積核大小)， 早期在每一層使用的activation function = 'tanh'。 對於手寫辨識來說一般的MLP效果就很不錯了，作者主要展示如何透過卷積降低網路的運算複雜度，與如何反向傳播。

上面的結構，只是一種參考，在現實使用中，每一層特徵圖需要多少個，卷積核大小選擇，還有池化的時候採樣率要多少，等這些都是變化的，這就是所謂的CNN調參，我們需要學會靈活多變。

比如我們可以把上面的結構改為:C1層卷積核大小為7*7，然後把C3層卷積核大小改為3*3等，然後特徵圖的個數也是自己選，說不定得到手寫字體識別的精度比上面那個還高，這也是有可能的，總之一句話：需要學會靈活多變，需要學會CNN的調參。

```Python

def LeNet(out):

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(6, kernel_size=5, strides=1, activation='tanh'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Conv2D(16, kernel_size=5, strides=1, activation='tanh'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Flatten())
        net.add(nn.Dense(120, activation='tanh'))
        net.add(nn.Dense(84, activation='tanh'))
        net.add(nn.Dense(out))

    return net
	
```

### feature map

什麼叫特徵圖呢？特徵圖其實說白了就是CNN中的每張圖片，都可以稱之為特徵圖。在CNN中，我們要訓練的卷積核並不是僅僅只有一個，這些卷積核用於提取特徵，卷積核個數越多，提取的特徵圖越多，

理論上來說精度也會更高，然而卷積核一堆，意味著我們要訓練的參數的個數越多。在LeNet-5經典結構中，第一層卷積核選擇了6個，而在AlexNet中，第一層卷積核就選擇了96個，具體多少個合適，還有待調整。

如何得到特徵圖呢？CNN的每一個卷積層我們都要人為的選取合適的卷積核個數，及卷積核大小。每個卷積核與輸入圖片進行卷積，就可以得到一張特徵圖。

比如LeNet-5經典結構中，第一層卷積核選擇了6個，所以我們可以得到6個特徵圖，這些特徵圖也就是下一層網絡的輸入。我們也可以把這些輸入圖片看成一張多通道特徵圖，作為第二層的輸入。


# 使用方法

```Python

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
net = le.LeNet()
net.initialize(init.Xavier())

epochs = 5*20
lr = 0.03
wd = 0.001
batch_size = 64
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'wd': wd})


net.load_parameters('lenet.params')  #若第一次執行請註解此行
train(epochs, batch_size, net, loss, trainer, train_data,transformer)
net.save_parameters('lenet.params')

```

# 結果

```Python 

acc:  0.9112
epoch: 100, loss: 0.166341, time: 21.501633 sec

```


# 原文出處

https://blog.csdn.net/d5224/article/details/68928083

http://zh.gluon.ai/chapter_convolutional-neural-networks/lenet.html

