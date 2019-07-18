# 前言

在LeNet提出後的將近20年裡，神經網絡一度被其他機器學習方法超越，如支持向量機。雖然LeNet可以在早期的小數據集上取得好的成績，但是在更大的真實數據集上的表現並不盡如人意。

一方面，神經網絡計算複雜。雖然20世紀90年代也有過一些針對神經網絡的加速硬件，但並沒有像之後GPU那樣大量普及。因此，訓練一個多通道、多層和有大量參數的捲積神經網絡在當年很難完成。

另一方面，當年研究者還沒有大量深入研究參數初始化和非凸優化算法等諸多領域，導致複雜的神經網絡的訓練通常較困難。

我們在上一節看到，神經網絡可以直接基於圖像的原始像素進行分類。這種稱為端到端（end-to-end）的方法節省了很多中間步驟。

然而，在很長一段時間裡更流行的是研究者通過勤勞與智慧所設計並生成的手工特徵。這類圖像分類研究的主要流程是：

* 獲取圖像數據集；
* 使用已有的特徵提取函數生成圖像的特徵；
* 使用機器學習模型對圖像的特徵分類。

在當時，機器學習盛行的程度遠高於類神經網路，機器學習優雅的定理證明了許多分類器的性質。機器學習領域生機勃勃、嚴謹而且極其有用。支持向量機(SVM)甚至被譽為人工智慧最頂尖的代表。

但是如果跟電腦視覺研究者交談，他們會告訴你圖像識別裡的現實是：** 計算機視覺流程中真正重要的是數據和特徵 ** 。
 
## 學習特徵表示

既然特徵如此重要，它該如何表示呢？

我們已經提到，在相當長的時間裡，特徵都是基於各式各樣手工設計的函數從數據中提取的。事實上，不少研究者通過提出新的特徵提取函數不斷改進圖像分類結果。這一度為計算機視覺的發展做出了重要貢獻。

然而，另一些研究者則持異議。他們認為特徵本身也應該由學習得來。他們還相信，為了表徵足夠複雜的輸入，特徵本身應該分級表示。持這一想法的研究者相信，多層神經網絡可能可以學得 ** 數據的多級表徵 **，並逐級表示越來越抽象的概念或模式。

以圖像分類為例，在多層神經網絡中，圖像的第一級的表示可以是在特定的位置和⻆度是否出現邊緣；而第二級的表示說不定能夠將這些邊緣組合出有趣的模式，如花紋；在第三級的表示中，也許上一級的花紋能進一步匯合成對應物體特定部位的模式。

這樣逐級表示下去，最終，模型能夠較容易根據最後一級的表示完成分類任務。需要強調的是，輸入的逐級表示由多層模型中的參數決定，而這些參數都是學出來的。


# AlexNet

2012年，AlexNet橫空出世。這個模型的名字來源於論文第一作者的姓名Alex Krizhevsky [1]。

AlexNet使用了8層卷積神經網絡，並以很大的優勢贏得了ImageNet 2012圖像識別挑戰賽。它首次證明了學習到的特徵可以超越手工設計的特徵，從而一舉打破計算機視覺研究的前狀，之後 ImageNet 的冠軍一直是 CNN。

AlexNet與LeNet的設計理念非常相似，但也有顯著的區別。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/alexnet.png)

```Python

def AlexNet(out):

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=3, strides=2))
        net.add(nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=3, strides=2))
        net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'))
        net.add(nn.MaxPool2D(pool_size=3, strides=2))
        net.add(nn.Dense(4096, activation="relu"))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(4096, activation="relu"))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(out))

    return net


```


### 模型複雜度

與相對較小的LeNet相比(訓練時間多了7倍)，AlexNet包含8層變換，其中有5層卷積和2層全連接隱藏層，以及1個全連接輸出層。在當時已經算是踏入層數加深的深度學習領域並證明CNN層數加深能帶來更好的結果。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/deep.png)


## Activation Function：使用 ReLU

當時最常用的激活函數（Activation Function）是 Sigmoid 和 tanh function。

AlexNet將sigmoid激活函數改成了更加簡單的ReLU激活函數。一方面，ReLU激活函數的計算更簡​​單，例如它並沒有sigmoid激活函數中的求冪運算。另一方面，ReLU激活函數在不同的參數初始化方法下使模型更容易訓練。

這是由於當sigmoid激活函數輸出極接近0或1時，這些區域的梯度幾乎為0，從而造成反向傳播無法繼續更新部分模型參數；而ReLU激活函數在正區間的梯度恆為1。

因此，若模型參數初始化不當，sigmoid函數可能在正區間得到幾乎為0的梯度，從而令模型無法得到有效訓練。，具體說明參閱activation.md



## 降低 Overfitting 的方法


### Dropout

在 AlexNet 中，第六層和第七層的全連階層使用 Dropout，配置為 0.5，表示每個神經元有 50% 的機率不參與下一層的傳遞。

由於在訓練中隱藏層神經元的丟棄是隨機的，即每個神經元都有可能被清零，輸出層的計算無法過度依賴任一個，從而在訓練模型時起到正則化的作用，並可以用來應對過擬合。

注: dropout 中的丟棄式透過訓練前隨機將神經元權重修改為 0， 當模型訓練完畢後，我們會設計開關讓dropout不會發生作用，在測試模型時，拿到更加穩定的結果。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/dropout.png)

### Data augmentation(圖像增廣)

AlexNet 對原本圖像做了處理後加入數據集中擴大數據集大小，將資料擴增 2048 倍。論文中說明此作法能有效的避免 Overfitting。

這是一個挺時用的技巧，當我們數據集大小不夠時，我們可以對原始圖像進行如翻轉、裁剪和顏色變化，從而進一步擴大數據集來緩解過擬合。

* 原始圖片的像素是 256 * 256，進行隨機抽取其中的 224 * 224 ，且允許水平翻轉
*  RGB 色彩空間做主成份分析（PCA），接著用高斯隨機擾動產生不同RGB比例的圖像。這個方法是透過自然圖片的性質來實現，也就是該物體對於照明的強度和顏色的變化是不變的。透過這個方法，top-1 的錯誤率下降 1%。


## 其他

百萬大小數據集、GPU運算的支援都是讓AlexNet如此成功的因素之一，也讓自LeNet之後逐漸沒落的類神經網路重新崛起。



# CIFAR 測試


### 下載數據集

```Python

train_data = gdata.vision.CIFAR10(train=True)
test_data = gdata.vision.CIFAR10(train=False)

```

### 訓練

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
		
		
transformer = gdata.vision.transforms.ToTensor()
net = alex.TinyAlexNet(10)
net.initialize(init.Xavier())

epochs = 10
lr = 0.03
wd = 0.001
batch_size = 64
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'wd': wd})


net.load_parameters('TinyAlex.params')
train(epochs, batch_size, net, loss, trainer, train_data,transformer)
net.save_parameters('TinyAlex.params')

```

# 結果

```Python

train_data = gdata.vision.CIFAR10(train=True)
test_data = gdata.vision.CIFAR10(train=False)

```



# 原文出處

http://zh.gluon.ai/chapter_convolutional-neural-networks/alexnet.html

https://medium.com/@WhoYoung99/alexnet-%E6%9E%B6%E6%A7%8B%E6%A6%82%E8%BF%B0-988113c06b4b

https://ithelp.ithome.com.tw/articles/10205088

[1] Krizhevsky, A., Sutskever, I., & Hinton, GE (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).






