


# 前言

## Siamese network 孿生神經網絡--一個簡單神奇的結構

十九世紀泰國出生了一對連體嬰兒，當時的醫學技術無法使兩人分離出來，於是兩人頑強地生活了一生，1829年被英國商人發現，進入馬戲團，在全世界各地表演，

1839年他們訪問美國北卡羅萊那州後來成為“ 玲玲馬戲團 ”的台柱，最後成為美國公民。1843年4月13日跟英國一對姐妹結婚，恩生了10個小孩，昌生了12個，姐妹吵架時，兄弟就要輪流到每個老婆家住三天。

1874年恩因肺病去世，另一位不久也去世，兩人均於63歲離開人間。兩人的肝至今仍保存在費城的馬特博物館內。從此之後“ 暹羅雙胞胎 ”（Siamese twins）就成了連體人的代名詞，也因為這對雙胞胎讓全世界都重視到這項特殊疾病。


![](https://github.com/rockuass1235/deep-learning/blob/master/images/twins.jpg)


簡單來說，Siamese network就是“連體的神經網絡”，神經網絡的“ 連體”是通過共享權值來實現的，如下圖所示。

在代碼實現的時候，甚至可以是同一個網絡，不用實現另外一個，因為權值都一樣。對於siamese network，兩邊可以是lstm或者cnn，都可以。


![](https://github.com/rockuass1235/deep-learning/blob/master/images/twins_network.jpg)


## 孿生神經網絡的用途是什麼？

簡單來說，衡量兩個輸入的相似程度。孿生神經網絡有兩個輸入（Input1 and Input2）,將兩個輸入feed進入兩個神經網絡（Network1 and Network2），這兩個神經網絡分別將輸入映射到新的空間，形成輸入在新的空間中的表示。

**通過Loss的計算，評價兩個輸入的相似度** 。

EX: 簽名筆跡驗證、 人臉辨識

先上結論：孿生神經網絡用於處理兩個輸入"比較類似"的情況。偽孿生神經網絡適用於處理兩個輸入"有一定差別"的情況。比如，我們要計算兩個句子或者詞彙的語義相似度，使用siamese network比較適合；

如果驗證標題與正文的描述是否一致（標題和正文長度差別很大），或者文字是否描述了一幅圖片（一個是圖片，一個是文字），就應該使用pseudo-siamese network。

也就是說，要根據具體的應用，判斷應該使用哪一種結構，哪一種Loss。





# Triplet Loss

Triplet loss最初是在FaceNet: A Unified Embedding for Face Recognition and Clustering論文中提出的，可以學到較好的人臉的embedding

輸入是一個三元組 <a, p, n>

* a： anchor
* p： positive,與a是同一類別的樣本
* n： negative,與a是不同類別的樣本


![](https://github.com/rockuass1235/deep-learning/blob/master/images/triplet_loss.png)



## 公式:

按照特徵向量的特性，一張影像所擷取的特徵向量應該要與其他相同內容的影像取得較為接近的特徵向量，與自己不相符的的影像所擷取的特徵向量盡可能差異化。

透過這樣的關係我們可以歸納出以下的公式，其中的margin是閥值，margin越大 distance(a,p)-distance(a, n)就要越小。 也就是說 distance(a,p) << distance(a, n)


Loss = max( distance(a, p) - distance(a, n) + margin, 0)
	
![](https://github.com/rockuass1235/deep-learning/blob/master/images/triplet_loss_formula.png)

所以最終目標就是拉近 (a, p)之間的距離， 拉遠(a, n)之間的距離


![](https://github.com/rockuass1235/deep-learning/blob/master/images/triplets_data.png)

訓練上資料有3種狀況:


* easy triplets: distance(a, p) + margin <  distance(a, n)

* semi-hard triplets: distance(a, p) <  distance(a, n) < distance(a, p) + margin

* hard triplets: distance(a, p) >  distance(a, n)

原則上除了easy triplets的情況不進行訓練， 所以Loss設為0




# 實作 Triplet 數據集


```Python

class TripletDataset(gdata.Dataset):

    def __init__(self, X, Y):

        if not isinstance(X, nd.NDArray):
            raise Exception('type of X is not nd.NDArray')
        if not isinstance(Y, nd.NDArray):
            raise Exception('type of Y is not nd.NDArray')

        self.X = X
        self.Y = Y
        self.classes = np.unique(self.Y.asnumpy())
        self.cls_num = len(self.classes)
        self.groups = tuple(
            np.where(Y.asnumpy() == cls)[0] for cls in self.classes)  # np.where return 座標(tuple)故取idx[0]
        self.pairs = self._get_pairs()

    def __getitem__(self, idx):

        idx = self.pairs[idx]

        return self.X[idx][0], self.X[idx][1], self.X[idx][2]

    def _get_pairs(self):

        pairs = []
        for i, indeces in enumerate(self.groups):

            np.random.shuffle(indeces)

            for j in range(len(indeces) - 1):
                piv = indeces[j]
                pos = indeces[j + 1]

                neg = i + random.randint(1, self.cls_num - 1)
                neg %= self.cls_num
                neg_idx = random.randint(0, len(self.groups[neg]) - 1)  # a <= random <= b
                neg = self.groups[neg][neg_idx]

                pairs.append(nd.array([piv, pos, neg]))

        return pairs

    def __len__(self):

        return len(self.pairs)

    def transform_all(self, fn, lazy=True):

        return self.transform(_TransformAllClosure(fn), lazy)
		

```

![](https://github.com/rockuass1235/deep-learning/blob/master/images/triplet_face.png)


# Model

使用ResNet18 v2架構抽取特徵映射至128D

```
ResNetV2(
  (features): HybridSequential(
    (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=True, use_global_stats=False, in_channels=3)
    (1): Conv2D(3 -> 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
    (3): Activation(relu)
    (4): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False)
    (5): HybridSequential(
      (0): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
        (conv1): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
        (conv2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
      (1): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
        (conv1): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
        (conv2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    (6): HybridSequential(
      (0): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
        (conv1): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
        (conv2): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (downsample): Conv2D(64 -> 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      )
      (1): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
        (conv1): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
        (conv2): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    (7): HybridSequential(
      (0): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)
        (conv1): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
        (conv2): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (downsample): Conv2D(128 -> 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      )
      (1): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
        (conv1): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
        (conv2): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    (8): HybridSequential(
      (0): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)
        (conv1): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
        (conv2): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (downsample): Conv2D(256 -> 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
      )
      (1): BasicBlockV2(
        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
        (conv1): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
        (conv2): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    (9): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)
    (10): Activation(relu)
    (11): GlobalAvgPool2D(size=(1, 1), stride=(1, 1), padding=(0, 0), ceil_mode=True)
    (12): Flatten
  )
  (output): Dense(512 -> 128, Activation(sigmoid))
)
```

## 評估函數

```Python

def evaluate_net(model, test_iter, ctx):
    
    loss = gluon.loss.TripletLoss(margin=0)
    sum_correct = 0
    sum_all = 0
    rate = 0.0
    
    for a, b, c in test_iter:
        
        a = a.as_in_context(ctx)
        b = b.as_in_context(ctx)
        c = c.as_in_context(ctx)
        
            
        anchor = net(a)
        positive = net(b)
        negative = net(c)
                
        l = loss(anchor, positive, negative)

        l = l.asnumpy()
        
        n_all = l.shape[0]
        #print(np.where(l == 0, 1, 0))
        n_correct = np.sum(np.where(l == 0, 1, 0))

        sum_correct += n_correct
        sum_all += n_all
        rate = sum_correct/sum_all
    
    print('acccuracy: %f (%s / %s)' % (rate, sum_correct, sum_all))
    return rate            
	
```

## 分類函數 K-最近鄰

```
def nearestclass(x, y, threshold):

    d = y[:, 1:] - x
    
    d = d**2
    d = d**0.5
    
    d = d.sum(axis = 1)
    print(d)
    idx = nd.argmin(d, axis = 0)
    
    
    print(idx)
    if d[idx] > threshold:
        return '???'
    
    return y[idx, 0]
```


# 結果

分類結果很不好，大概是樣本數過少(300筆)

用MNIST測試(60000筆) 結果倒是挺不錯的

```

epochs: 41, loss: 0.000340, time: 7.840783 sec
acccuracy: 0.612903 (152 / 248)
epochs: 42, loss: 0.000321, time: 7.912023 sec
acccuracy: 0.608871 (151 / 248)
epochs: 43, loss: 0.000309, time: 7.747280 sec
acccuracy: 0.600806 (149 / 248)
epochs: 44, loss: 0.000281, time: 7.766007 sec
acccuracy: 0.600806 (149 / 248)
epochs: 45, loss: 0.000248, time: 7.674172 sec
acccuracy: 0.600806 (149 / 248)



```


	



# 資料來源


https://zhuanlan.zhihu.com/p/35040994

http://lawlite.me/2018/10/16/Triplet-Loss%E5%8E%9F%E7%90%86%E5%8F%8A%E5%85%B6%E5%AE%9E%E7%8E%B0/