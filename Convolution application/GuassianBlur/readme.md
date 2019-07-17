# 模糊化 Blur

所謂”模糊”，可以理解成每一個像素都取周邊像素的平均值。

中間點取周圍點的平均值，在數值上，這是一種 ** 平滑化 ** 。在圖形上，就相當於產生模糊效果，中間點失去細節。 計算平均值時，取值範圍越大，模糊效果越強烈。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/avg_blur.png)


# 高斯模糊

如果使用簡單平均，顯然不是很合理，因為圖像都是連續的，越靠近的點關係越密切，越遠離的點關係越疏遠。因此，加權平均更合理，距離越近的點權重越大，距離越遠的點權重越小。

所以我們可以對圖形用高斯正態分佈進行加權平均。

### 高斯2D

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ga_blur_2d.png)

### 高斯3D

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ga_blur_3d.png)



### 高斯公式

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ga_formula.png)


假定中心點的坐標是（0,0），那麼距離它最近的8個點的坐標如下：

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ga_pivot.png)


為了計算權重矩陣，需要設定σ的值。假定σ=1.5，則權重矩陣如下：

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ga_weight.png)

這9個點的權重總和等於0.4787147，如果只計算這9個點的加權平均，還必須讓它們的權重之和等於1，因此上面9個值還要分別除以總和，得到最終的權重矩陣。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ga_final_weight.png)


注: 如果圖像通道大於1 則對每一個通道個別做guassian blur後疊加。



```Python

def G(x, y, o):
    return math.exp(-(x**2 + y**2)/(2 * o**2)) / (2 * math.pi * o**2)


class GuassianFilter(nn.Block):

    def __init__(self, size, o = 1.5,**kwargs):
        super(GuassianFilter, self).__init__(**kwargs)
        self.size = size
        self.o = o
        self.blk = nn.Conv2D(1, kernel_size=size, padding=(self.size-1)//2)

    def initialize(self, ctx=None, verbose=False, force_reinit=False):
        super(GuassianFilter, self).initialize()
        mid = (self.size-1)//2

        k = nd.zeros(shape=(1, 1, self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                k[:, :, i, j] = G(j-mid, mid-i, 1.5)
        k = k / k.sum().asscalar()
        self.blk.weight.set_data(k)

    def forward2d(self, x):
        shape = x.shape
        x = x.reshape((1, 1, *x.shape))
        x = x.astype('float32')
        x = self.blk(x)
        x.astype('uint8')
        return x.reshape((shape[0], shape[1]))

    def forward(self, x):

        if len(x.shape) < 3:
            return self.forward2d(x)
        else:
            yhat = nd.zeros(shape=(x.shape[2], x.shape[0], x.shape[1]))
            x = x.transpose((2,0,1))

            for y, x in zip(yhat, x):
                y[:] = self.forward2d(x)
            return yhat.transpose((1,2,0))
```


# 原文出處

https://blog.csdn.net/nima1994/article/details/79776802


