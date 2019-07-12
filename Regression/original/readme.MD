<h1>線性回歸</h1>
線性回歸輸出是一個連續值，因此適用於回歸問題。回歸問題在實際中很常見，如預測房屋價格、氣溫、銷售額等連續值的問題。與回歸問題不同，分類問題中模型的最終輸出是一個離散值。我們所說的圖像分類、垃圾郵件識別、疾病檢測等輸出為離散值的問題都屬於分類問題的範疇。softmax回歸則適用於分類問題。

由於線性回歸和softmax回歸都是單層神經網絡，它們涉及的概念和技術同樣適用於大多數的深度學習模型。我們首先以線性回歸為例，介紹大多數深度學習模型的基本要素和表示方法。<br>

<h1>讀取數據</h1>
在訓練模型的時候，我們需要遍歷數據集並不斷讀取小批量數據樣本。這裡我們定義一個函數：它每次返回batch_size（批量大小）個隨機樣本的特徵和標籤。
```Python
def data_iter(X, Y, batch_size):

    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    for i in range(0, n, batch_size):
        j = nd.array(indices[i:i+batch_size])
        yield X.take(j), Y.take(j)

```

<h1>模型</h1>
設房屋的面積為 x1 ，房齡為 x2 ，售出價格為 y 。我們需要建立基於輸入 x1 和 x2 來計算輸出 y 的表達式，也就是模型（model）。
顧名思義，線性回歸假設輸出與各個輸入之間是線性關係：<br>

##nd.dot(X,W) + b<br>
```Python
class Layer(object):

    def __init__(self, intput, output):
        self.w = nd.random.normal(scale=0.01, shape=(intput, output))
        self.b = nd.zeros(shape=(output,))
        self.w.attach_grad()
        self.b.attach_grad()

    def __call__(self, x):
        return nd.dot(x,self.w) + self.b
        
class Net(object):

    def __init__(self):
        self.__layer = []
    def add(self, layer):

        self.__layer.append(layer)
    def all_params(self):

        params = []
        for l in self.__layer:
            params.append(l.w)
            params.append(l.b)
        return params
    def __call__(self, x):

        for  l in self.__layer:
            x = l(x)
        return x


```
其中 W 是權重（weight）， b 是偏差（bias），且均為標量。它們是線性回歸模型的參數（parameter）。模型輸出 y^ 是線性回歸對真實價格 y 的預測或估計。我們通常允許它們之間有一定誤差。

<h1>損失函數</h1>
  在模型訓練中，我們需要衡量價格預測值與真實值之間的誤差。通常我們會選取一個非負數作為誤差，且數值越小表示誤差越小。一個常用的選擇是平方函數。<br>

```Python
def l2loss(yhat, y):
  return (yhat-y.reshape(yhat.shape))**2/2
```
其中常數 1/2 使對平方項求導後的常數係數為1，這樣在形式上稍微簡單一些。顯然，誤差越小表示預測價格與真實價格越相近，且當二者相等時誤差為0。給定訓練數據集，這個誤差只與模型參數相關，因此我們將它記為以模型參數為參數的函數。在機器學習裡，將衡量誤差的函數稱為損失函數（loss function）。這裡使用的平方誤差函數也稱為平方損失（square loss）。<br>

在模型訓練中，我們希望找出一組模型參數，記為 w*, b*  ，來使訓練樣本平均損失最小

<h1>優化算法</h1>
 當模型和損失函數形式較為簡單時，上面的誤差最小化問題的解可以直接用公式表達出來。這類解叫作解析解（analytical solution）。本節使用的線性回歸和平方誤差剛好屬於這個範疇。然而，大多數深度學習模型並沒有解析解，只能通過優化算法有限次迭代模型參數來盡可能降低損失函數的值。這類解叫作數值解（numerical solution）。<br>
  
 在求數值解的優化算法中，小批量隨機梯度下降SGD（mini-batch stochastic gradient descent）在深度學習中被廣泛使用。它的算法很簡單：先選取一組模型參數的初始值，如隨機選取；接下來對參數進行多次迭代，使每次迭代都透過反向求導嘗試降低損失函數的值。在每次迭代中，先隨機均勻採樣一個由固定數目訓練數據樣本所組成的小批量（mini-batch） B ，然後求小批量中數據樣本的平均損失有關模型參數的導數（梯度），最後用此結果與預先設定的一個正數的乘積作為模型參數在本次迭代的減小量。<br>
 
 ```Python
 def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr/batch_size * param.grad
        
```

<h1>模型預測</h1>
模型訓練完成後，這裡我們得到的並不一定是最小化損失函數的最優解 w∗1,w∗2,b∗ ，而是對最優解的一個近似。然後，我們就可以使用學出的線性回歸模型 x1w^1+x2w^2+b^ 來估算訓練數據集以外任意一棟面積（平方米）為 x1 、房齡（年）為 x2 的房屋的價格了。這裡的估算也叫作模型預測、模型推斷或模型測試。<br>




 
 
