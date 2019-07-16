# 轉灰階濾波器


對於彩色轉灰度，有一個很著名的心理學公式：

Gray = R*0.299 + G*0.587 + B*0.114

而實際應用時，希望避免低速的浮點運算，所以需要整數算法。

Gray = (R*299 + G*587 + B*114 + 500) / 1000

上面的整數算法已經很快了，但是有一點仍制約速度，就是最後的那個除法。移位比除法快多了，所以可以將係數縮放成2的整數冪，而我則是使用張量計算取代位移法。

mxnet的convolution 接受的資料樣式一定要 shape = (batch_size, channel, h, w)，所以我override 它的forward() 讓他接受img時自動做一個轉換格式。



```Python
class GrayFilter(nn.Block):

    def __init__(self, **kwargs):
        super(GrayFilter, self).__init__(**kwargs)
        self.blk = nn.Conv2D(1, 1)

    def initialize(self, ctx=None, verbose=False,force_reinit=False):
        super(GrayFilter, self).initialize()
        k = [[[[299]], [[587]], [[114]]]]
        k = nd.array(k)
        b = nd.ones(shape=(1,))*500
        self.blk.weight.set_data(k)
        self.blk.bias.set_data(b)


    def forward(self, x):
        shape = x.shape
        x = x.reshape((1, *x.shape))
        x = x.transpose((0, 3, 1, 2))
        x = x.astype('float32')
        x = self.blk(x)
        x = x/1000
        x.astype('uint8')
        return x.reshape((shape[0], shape[1]))
		
```

# 使用方式

```Python

net = GrayLayer()
net.initialize()

X = image.imread('lena.jpg')
yhat = net(X)
yhat = yhat.astype('uint8')

plt.imshow(yhat.asnumpy(), plt.cm.gray)
plt.show()


```

# 結果


### 原圖

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/lena.jpg)

### 灰階

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/gray.png)




#原文出處

http://atlaboratary.blogspot.com/2013/08/rgb-g-rey-l-gray-r0.html




