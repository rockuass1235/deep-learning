# 實戰Kaggle比賽：房價預測


## Kaggle
Kaggle是一個著名的供機器學習愛好者交流的平台。圖3.7展示了Kaggle網站的首頁。為了便於提交結果，需要註冊Kaggle賬號。

https://www.kaggle.com/

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/kaggle.png)

我們可以在房價預測比賽的網頁上了解比賽信息和參賽者成績，也可以下載數據集並提交自己的預測結果。

該比賽的網頁地址是 https://www.kaggle.com/c/house-prices-advanced-regression-techniques



## 資料處理

pandas 是 Python提供的關於操作資料的函式庫，它提供函式讀取csv檔案，並讓資料可以如同操作mysql一般輸出資料

比賽數據分為訓練數據集和測試數據集。兩個數據集都包括每棟房子的特徵，如街道類型、建造年份、房頂類型、地下室狀況等特徵值。這些特徵值有連續的數字、離散的標籤甚至是缺失值“na”。
只有訓練數據集包括了每棟房子的價格，也就是標籤。

下面使用pandas讀取這兩個文件。

```Python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```

為了讓資料在空間的散佈狀況更好，也為了統一特徵間的尺度，我們對資料做標準化

我們對連續數值的特徵做標準化（standardization）：設該特徵在整個數據集上的均值為 μ ，標準差為 σ 。
那麼，我們可以將該特徵的每個值先減去 μ 再除以 σ 得到標準化後的每個特徵值。

```Python

features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
num_idx = features.dtypes[features.dtypes != 'object'].index
features[num_idx] = features[num_idx].apply(lambda x: (x - x.mean()) / (x.std()))

```

我們對於資料缺失向(NA)，對其補上該欄的平均值，避免缺失項影響訓練情況

```Python
features[num_idx] = features[num_idx].fillna(0)
```

在處理完數值型資料後，我們接著處理文字型資料。我們將文字換成one-hot 離散型特徵值
EX:
|環境類型|房頂類型|地下室|
|---|---|---|
|郊區|鐵皮|無|
|小鎮|歐式|有|

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
## 效果
![image](https://github.com/rockuass1235/deep-learning/blob/master/images/fold1.png)
![image](https://github.com/rockuass1235/deep-learning/blob/master/images/fold2.png)
![image](https://github.com/rockuass1235/deep-learning/blob/master/images/fold3.png)