# 讀入數據
我們通過Gluon的data包來下載這個數據集。第一次調用時會自動從網上獲取數據。我們通過參數train來指定獲取訓練數據集或測試數據集（testing data set）。測試數據集也叫測試集（testing set），只用來評價模型的表現，並不用來訓練模型。

```Python
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
```


# 顯示數據資料

```Python
def show_data(images, labels):

    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
```
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/mnist.PNG)

#   資料預處理
了解數據不是數據科學中最困難的事情，但這的確是一件非常耗時的事情。很多人可能會忽略這一步驟，就直接下水了。

1. 理解問題：我們將研究每個變量，並對這個問題的意義和重要性進行哲學分析。

2. 單變量研究：我們只關注因變量（'SalePrice'）並嘗試更多地了解它。

3. 多變量研究：我們將嘗試了解因變量和自變量之間的關係。

4. 基本的清理工作：我們將清理數據集並處理缺失的數據，異常值和分類變量。

5. 測試假設：我們將檢查我們的數據是否符合大多數多元技術所需的假設。
#  數值標準化處理

###  數據縮放的本質是什麼:
數據normalize or standardize後，最優解的尋優過程明顯會變得平緩，更容易正確   的收斂到最優解。

###  不同數據縮放的區別
引入歸一化，是由於在不同評價指標（特徵指標）中，其單位往往不同，變化區間處於不同的數量級，若不進行歸一化，可能導致某些指標被忽視，影響到數據分析的結果。(1 在0-1之間是最大， 在0-10000之間反而很微小)
為了消除特徵數據之間的量綱影響，需要進行歸一化處理，以解決特徵指標之間的可比性。原始數據經過歸一化處理後，各指標處於同一數量級，以便進行綜合對比評價。

###  如何選擇適合的縮放方法
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/normalize.png)

數據normalize or standardize後，最優解的尋優過程明顯會變得平緩，更容易正確   的收斂到最優解。
#####   未歸一化求導
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/nonormal.png)
#####   歸一化求導
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/normal.png)

#   建立模型
```Python
net = nn.Sequential()
net.add(nn.Dense(256, activation = 'relu'))
net.add(nn.Dense(256 , activation= 'relu'))
net.add(nn.Dense(10))
net.initialize(init = init.Xavier())
```
