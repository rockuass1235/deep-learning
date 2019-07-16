
# Convolution implement
我們依照Convolution 的定義實作卷積動作，然而實際上對於每個元素賦值的動作很慢，比起張量計算慢很多

而且大部分的框架對於非張量計算不提供自動求導功能

```Python
def corr2d(X, k):

    h,w = k.shape

    shape = X.shape[0]-h+1, X.shape[1]-w+1
    Y = nd.zeros(shape)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i:i+h, j:j+w]*k).sum().asscalar()
    return Y
```


本次將在內容裡實作一些卷積對於圖像的應用，為了加速運算會直接使用mxnet提供的 gluon.nn.Conv2D()來實作

EX:　nn.Conv2D(1, kernel_size=(1, 2))

