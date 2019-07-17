# Adaptive Threshold (自適應2值化濾波器)

2值化在圖像處理中是很常用的一種手法，我們可以透過直方圖(Histogram)分析找出恰當的閥值將圖像2值化。

但是在一些場合，並不能得到較好的結果，此時adaptive threshold也許能得到一個不錯的結果。

# 作法

我們利用卷積核滑動找出區域面積的平均值與中心像素點做比較，若中心點大於平均值則輸出1 反之輸出 0


### 原始中心值 卷積核

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/original_kernal.png)

### 平均值 卷積核

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/avg_kernal.png)

### 原始-平均值

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/final_kernal.png)


```Python

class AdaptiveThreshold(nn.Block):

    def __init__(self,size = 3,  **kwargs):
        super(AdaptiveThreshold, self).__init__(**kwargs)
        self.size = size
        self.blk = nn.Conv2D(1, size, padding=(self.size-1)//2)


    def initialize(self, ctx=None, verbose=False, force_reinit=False):
        super(AdaptiveThreshold, self).initialize()
        r = 1/self.size**2
        k = nd.ones(shape=(1,1,self.size, self.size)) * (-r)
        k[0,0,(self.size-1)//2,(self.size-1)//2] += 1
        self.blk.weight.set_data(k)

    def forward(self, x):
        shape = x.shape
        x = x.reshape((1,1,*x.shape))
        x = x.astype('float32')
        x = self.blk(x)
        x = x > 0
        x = x*255
        x.astype('uint8')
        return x.reshape((shape[0], shape[1]))


```


# 使用方法

```Python

net = nn.Sequential()
net.add(gray.GrayFilter())
net.add(AdaptiveThreshold(size= 63))

for layer in net:
    layer.initialize()

X = image.imread('sudoku-original.jpg')
yhat = net(X)
yhat = yhat.astype('uint8')



plt.imshow(yhat.asnumpy(), plt.cm.gray)
plt.show()

```

# 結果

### 原圖

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sukodu_original.jpg)

### AdaptiveThreshold

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sukodu_adaptive.png)

# 原文出處

None