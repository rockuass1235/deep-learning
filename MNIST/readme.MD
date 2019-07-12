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
