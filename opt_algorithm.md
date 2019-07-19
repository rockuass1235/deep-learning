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


# gradient descent 梯度下降法(GD)

我們都知道在一階導數為0的地方會有極值或奇異點存在，這也是設計loss function 而不是直接使用 accuracy function的原因之一。 

透過這樣的特性，設計一個 **凹向上的函數(又稱凸函數)** 作為loss function，當x 沿著梯度(斜率)反方向移動，會讓目標函數值越小。 所以Optimization Algorithm又有人叫凸優化演算法。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/fun.png)

給定絕對值足夠小的數 ϵ ，根據泰勒展開公式，我們也可以得到以下的近似：

<h1>f(x+ϵ)≈f(x)+ϵf′(x).</h1>




















# 原文出處

http://zh.gluon.ai/chapter_optimization/optimization-intro.html



