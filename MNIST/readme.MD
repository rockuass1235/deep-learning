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
#   WHY
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/%E5%9C%96%E7%89%871.png)
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/%E5%9C%96%E7%89%872.png)
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/%E5%9C%96%E7%89%873.png)
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/%E5%9C%96%E7%89%874.png)
##  程式碼
#### 資料缺失處理(NA)
對於缺失的特徵值，我們將其替換成該特徵的均值。<br>
data = data.fillna(data.mean())

#### Label Encoding & One-Hot Encoding
Label encoding在某些情況下很有用，但是場景限制很多。比如有一列[dog,cat,dog,mouse,cat]，我們把其轉換為[1,2,1,3,2]。這裡就產生了一個奇怪的現象：dog和mouse的平均值是cat。
One-Hot編碼，又稱為一位有效編碼，主要是採用位狀態寄存器來對個狀態進行編碼，每個狀態都由他獨立的寄存器位，並且在任意時候只有一位有效。
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/onehot.png)
#### 為什麼使用one-hot編碼來處理離散型特徵?

1. 使用one-hot編碼，將離散特徵的取值擴展到了歐式空間，離散特徵的某個取值就對應歐式空間的某個點。

2.將離散特徵通過one-hot編碼映射到歐式空間，是因為，在回歸，分類，聚類等機器學習算法中，特徵之間距離的計算或相似度的計算是非常重要的，而我們常用的距離或相似度的計算都是在歐式空間的相似度計算，計算餘弦相似性，基於的就是歐式空間。

3. 對離散型特徵進行one-hot編碼是為了讓距離的計算顯得更加合理。

4. 將離散型特徵進行one-hot編碼的作用，是為了讓距離計算更合理，但如果特徵是離散的，並且不用one-hot編碼就可以很合理的計算出距離，那麼就沒必要進行one -hot編碼

EX: 該離散特徵共有1000個值，我們分成兩組，分別是400和600,  兩個小組之間的距離有合適的定義，組內的距離也有合適的定義，那就沒必要用one-hot編碼



```Python
def standardize(x):
    return (x-x.mean())/x.std()

def numeric_standardize(data):
    
    idx = data.dtypes[data.dtypes != 'object'].index
    data[idx] = data[idx].apply(standardize)
    data = data.fillna(data.mean())
    return data

```

#   K-fold交叉驗證

在K-fold交叉驗證中我們訓練K次並返回訓練和驗證的平均誤差。
其作法:

1. 將資料分成K份，每一次將其中一份選為validation set，其他k-1份為train set
2. Train set 進行訓練模型
3. Validation set 進行驗證測試誤差
4. 重複1 – 3 步驟K次，並計算K次的平均驗證誤差
```Python
def get_k_fold_data(k, i, x, y):
    
    n = x.shape[0]
    train_x, train_y = None, None
    valid_x, valid_y = None, None
    for j in range(k):
        idx = slice(j * n // k, (j+1) * n // k)
        fold_x, fold_y = x[idx], y[idx]
        
        if j == i:
            valid_x, valid_y = fold_x, fold_y
        elif train_x is None:
            train_x, train_y = fold_x, fold_y
        else:
            train_x = nd.concat(train_x, fold_x, dim = 0)
            train_y = nd.concat(train_y, fold_y, dim = 0)
    
    return train_x, train_y, valid_x, valid_y

```
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/fold1.png)
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/fold2.png)
![image](https://github.com/rockuass1235/deep-learning/blob/master/MNIST/images/fold3.png)

#   建立模型
```Python
net = nn.Sequential()
net.add(nn.Dense(256, activation = 'relu'))
net.add(nn.Dense(256 , activation= 'relu'))
net.add(nn.Dense(10))
net.initialize(init = init.Xavier())
```
