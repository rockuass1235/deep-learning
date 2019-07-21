# 前言

在訓練模型時，我們會使用優化算法不斷迭代模型參數以降低模型損失函數的值。當迭代終止時，模型的訓練隨之終止，此時的模型參數就是模型通過訓練所學習到的參數。

優化算法對於深度學習十分重要。一方面，訓練一個複雜的深度學習模型可能需要數小時、數日，甚至數週時間，而優化算法的表現直接影響模型的訓練效率；

另一方面，理解各種優化算法的原理以及其中超參數的意義將有助於我們更有針對性地調參，從而使深度學習模型表現更好。

# Optimization Algorithm

深度學習中絕大多數目標函數都很複雜。因此，很多優化問題並不存在解析解，而需要使用基於數值方法的優化算法找到近似解，即數值解。

為了求得最小化 loss function 的數值解，我們將通過優化算法有限次迭代修正權重來盡可能降低損失函數的值。

在尋求數值解的過程中，我們找到的有可能是local minimum 或 critical point，要如何判斷是否為 global minimum或找出Optimization Algorithm的選擇能讓你事半功倍。



### Local Minimum

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/opt_min.svg)

### Critical point

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/opt_critical.svg)


# 梯度下降（gradient descent，GD)

我們都知道在一階導數為0的地方會有極值或奇異點存在，這也是設計loss function 而不是直接使用 accuracy function的原因之一。 

透過這樣的特性，設計一個 **凹向上的函數(又稱凸函數)** 作為loss function，當x 沿著梯度(斜率)反方向移動，會讓目標函數值越小。 所以Optimization Algorithm又有人叫凸優化演算法。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/fun.png)

## 證明

給定絕對值足夠小的數 ϵ ，根據泰勒展開公式，我們也可以得到以下的近似：

<h1>f(x+ϵ)≈f(x)+ϵf′(x).</h1>

若 ϵ = −ηf′(x) 帶入回得到以下公式:

<h1>f(x−ηf′(x))≈f(x)−ηf′(x)^2.</h1>

已知 f′(x)^2 > 0 所以:
<h1>f(x−ηf′(x))≲f(x).</h1>

得證 當x 沿著梯度(斜率)反方向移動，會讓目標函數值越小。 當x的維度 > 1 上面的證明依然成立

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sgd.svg)


### learning rate

上面公式的 η 通常叫作學習率。這是一個超參數，需要人工設定。如果使用過小的學習率，會導致 x 更新緩慢從而需要更多的迭代才能得到較好的解或者掉入local minimum中。

如果使用過大的學習率， |ηf′(x)| 可能會過大從而使前面提到的一階泰勒展開公式不再成立：這時我們無法保證迭代 x←x−ηf′(x) 會降低 f(x) 的值。



# 隨機梯度下降（stochastic gradient descent，SGD)

在深度學習裡，目標函數通常是訓練數據集中有關各個樣本的損失函數的平均。若數據集的大小為n，我們需要計算n筆資料取平均後才計算梯度。

假設n是一個天文數字(訓練資料通常都百萬起跳)，GD 需要進行n(百萬)次計算後才進行一次權重更新，無疑是非常沉重的成本。

為了加速權重更新的速度， 在統計學有一個概念，在母體中隨機抽樣產生的子樣本，子樣本的行為模式與原母體行為差異屬於無偏估計。

也就是我們可以設定一個batch size大小進行抽樣產生子樣本，子樣本所估計出來的梯度近似於用母體所估計出來的梯度。如此一來在每個batch size後，我們就能直接進行權重更新，增加更新頻率。

```Python 

lr = 0.03
trainer = gluon.Trainer(net.collect_params(), 'sgd',{'learning_rate': lr})

```

### 計算母體梯度更新

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sgd_all.svg)


### 計算批次樣本梯度更新

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sgd_batch.svg)




# 動量法(Momentum)

隨機梯度下降法(SGD)，是不敗的經典。即使在推出如此多的優化演算法，研究學者的論文大多數還是主要採用SGD做為優化演算法，然而SGD有一些問題很難克服:

### 落入local minimum:

在某些情況如果學習率太小，先不題訓練緩慢的問題，它有可能找不到真正的global minimum，而學習率設太大有可能導致無法收斂。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sgd_err1.png)

### 每個分量的斜率大小不一致:

假設資料X維度2維，可以看到，同一位置上，目標函數在豎直方向（ x2 軸方向）比在水平方向（ x1 軸方向）的斜率的絕對值更大。

因此，給定學習率太小， 有可能導致x1訓練緩慢甚至落入local minimum; 給定學習率太大，x2會無法收斂

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/mom_test.svg)


### Solution

動量法主要就是為了解決這個問題，降低對學習率的依賴。

原本變化 x←x−ηf′(x) 修正為: 

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/mom_formula.png)

動量超參數 γ 滿足 0≤γ<1 。當 γ=0 時，動量法等價於小批量隨機梯度下降。


我們可以看到如下圖:

當 學習率x斜率 很大時，v(動量)值反而因為反覆震盪而不會有所增長，動量超參數 γ讓其逐漸收斂; 

當 學習率x斜率 很小時，v(動量)值會同向疊加越來越大，加速收斂甚至給予其足夠動量離開 local minimum。





```Python 

mom = 0.9
trainer = gluon.Trainer(net.collect_params(), 'sgd',{'learning_rate': lr, 'momentum':mom})

```


# AdaGrad

adagrad 與 momentum 都是為了降低對於超參數學習率的依賴，如果說momentum對於平緩梯度的學習率增長速度較陡梯度快，那麼adagrad就與其相反，對於平緩梯度的學習率降低速度較陡梯度慢。

公式如下:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/adagrad_formula.png)

ϵ 是為了維持數值穩定性而添加的常數

由於 st 一直在累加按元素平方的梯度，自變量中每個元素的學習率在迭代過程中一直在降低（或不變）。所以，當學習率在迭代早期降得較快且當前解依然不佳時，AdaGrad算法在迭代後期由於學習率過小，可能較難找到一個有用的解。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/adagrad_show.jpg)

在參數空間更為平緩的方向，會取得更大的進步（因為平緩，所以歷史梯度平方和較小，對應學習下降的幅度較小），並且能夠使得陡峭的方向變得平緩，從而加快訓練速度。


```Python
trainer = gluon.Trainer(net.collect_params(), 'adagrad',{'learning_rate': lr})
							
```


# RMSProp

我們在“AdaGrad算法”一節中提到，因為調整學習率時分母上的變量 st 一直在累加按元素平方的小批量隨機梯度，所以目標函數自變量每個元素的學習率在迭代過程中一直在降低（或不變）。

因此，當學習率在迭代早期降得較快且當前解依然不佳時，AdaGrad算法在迭代後期由於學習率過小，可能較難找到一個有用的解。為了解決這一問題，RMSProp算法對AdaGrad算法做了一點小小的修改。

公式如下:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/rmsprop_formula.png)

ϵ 是為了維持數值穩定性而添加的常數

因為RMSProp算法的狀態變量 st 是對平方項 gt⊙gt 的指數加權移動平均，所以可以看作是最近 1/(1−γ) 個時間步的小批量隨機梯度平方項的加權平均。如此一來，自變量每個元素的學習率在迭代過程中就不再一直降低（或不變）。


```Python
trainer = gluon.Trainer(net.collect_params(), 'rmsprop', {'learning_rate': lr, 'gamma1': gamma})
							
```


# Adam

該算法名為「Adam」，其並不是首字母縮寫，也不是人名。它的名稱來源於適應性矩估計（adaptive moment estimation）。

原論文列舉了將Adam 優化算法應用在非凸優化問題中所獲得的優勢：

* 直截了當地實現高效的計算所需內存少梯度對角縮放的不變性（第二部分將給予證明） 適合解決含大規模數據和參數的優化問題適用於非穩態（non-stationary）目標

* 適用於解決包含很高噪聲或稀疏梯度的問題超參數可以很直觀地解釋，並且基本上只需極少量的調參

同樣在CS231n 課程中，Adam 算法也推薦作為默認的優化算法。

Adam算法是將Momentum算法和RMSProp算法結合起來使用的一種算法，一種可以使用類似於物理中的動量來累積梯度，另一種可以使得收斂速度更快同時使得波動的幅度更小。那麼講兩種算法結合起來所取得的表現一定會更好。

公式如下:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/adam_formula.png)


這裡對1階導數與2階導數都進行了指數加權移動平均

### 那什麼是指數加權平均?


指數移動平均（英語：exponential moving average，EMA或EWMA）是以指數式遞減加權的移動平均。各數值的加權影響力隨時間而指數式遞減，越近期的數據加權影響力越重，但較舊的數據也給予一定的加權值。

加權的程度以常數α決定，α數值介乎0至1。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ewma.png)


該算法是目前最重要的算法之一。從金融時間序列、信號處理到神經網絡，其應用非常廣泛。基本上所有的數據都是有序的。

我們主要使用此算法來減少噪音時間序列數據中的噪音。我們使用的術語是「smoothing」數據。

公式如下:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ewma_formula.jpg)


公式指出，t時刻的移動平均值(S)的值是t時刻原始信號(x)的值與移動平均本身的前值(即t-1)的混合。混合程度由參數a(0-1之間的值)控制。

所以，如果a = 10％（小），則大部分貢獻將來自信號的先前值。在這種情況下，「 smoothing 」將非常大。

如果a = 90％（大），則大部分貢獻將來自信號的當前值。在這種情況下，「smoothing 」將是小的。


### a = 0.1 進行EWMA

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/ewma_sample.jpg)


實際上 a 值也有一些實際的意義， 從數學上可以得知(1-1/n)^n = exp(-1) = 0.3679。假設exxp(-1)是很小的值，代表加權指數平均到某一定值以後可以視同0忽略不計

一般這個值 = 1/a， 假設a = 0.1，得到的值其實就是10輪的加權指數平均，離當前時間越近的值獲得的權重越大

注意: adagrad 變數a位置與ewma公式相反(β = 1-a)


```Python

 trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
							
```



# 原文出處

http://zh.gluon.ai/chapter_optimization/optimization-intro.html

http://zh.gluon.ai/chapter_optimization/adagrad.html

https://zhuanlan.zhihu.com/p/29920135

https://cloud.tencent.com/developer/article/1057062

https://kknews.cc/zh-tw/other/pg4xeq2.html

https://zh.wikipedia.org/wiki/%E7%A7%BB%E5%8B%95%E5%B9%B3%E5%9D%87







