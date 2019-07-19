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



### 計算母體梯度更新

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sgd_all.svg)


### 計算批次樣本梯度更新

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/sgd_batch.svg)










# 原文出處

http://zh.gluon.ai/chapter_optimization/optimization-intro.html



