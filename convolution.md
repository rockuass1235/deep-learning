# Convolution Layer

卷積神經網絡（convolutional neural network）是含有捲積層（convolutional layer）的神經網絡。最常見的二維卷積層。
它有高和寬兩個空間維度，常用來處理圖像數據。我們將介紹簡單形式的二維卷積層的工作原理。

在二維卷積層中，一個二維輸入數組和一個二維核（kernel）數組通過互相關運算輸出一個二維數組。我們用一個具體例子:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/conv.gif)

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/conv_formula.jpg)

它的物理意義大概可以理解為：系統某一時刻的輸出是由多個輸入共同作用（疊加）的結果。

卷積核上所有作用點依次作用於原始像素點後（即乘起來），線性疊加的輸出結果，即是最終卷積的輸出，也是我們想要的結果，我們稱為destination pixel.

（1）原始圖像通過與卷積核的數學運算，可以提取出圖像的某些指定特徵（features)。

（2）不同卷積核，提取的特徵也是不一樣的。

（3）提取的特徵一樣，不同的捲積核，效果也不一樣。

# Padding

當我們在用3x3 的捲積核在6x6 的圖像上執行卷積時，我們得到了4x4 的特徵圖。圖片變得縮小，假設我們補上邊使其成為8x8

再執行卷積時，特徵圖的大小就會變成6x6

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/conv_pad.svg)

圖像為padding示意圖，在周圍補0不影響類神經網路訓練(疊加0)

所以我們得到以下公式:　　![image](https://github.com/rockuass1235/deep-learning/blob/master/images/padding_formula.jpg)

如果我們希望影像大小經過卷積核處理後大小不變，在擴充的寬度(padding)應該滿足下面的方程，其中p 是padding（填充），f 是卷積核的維度（通常是奇數）。


# Stride

在之前的例子中，我們總是將捲積核移動一個像素。但是，步長也可以看做是卷積層的一個參數。我們可以看到，如果我們使用更大的步長，卷積會成為什麼樣子。

在設計CNN 結構時，如果我們想降低輸出矩陣大小，那麼我們可以決定增大步長。考慮到擴充(p)和跨步(s)，輸出矩陣的大小可以使用下面的公式計算：

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/stride_formula.jpg)


![image](https://github.com/rockuass1235/deep-learning/blob/master/images/stride.gif)


所以我們歸納以下結論:

* 填充可以增加輸出的高和寬。這常用來使輸出與輸入具有相同的高和寬。
* 步幅可以減小輸出的高和寬，例如輸出的高和寬僅為輸入的高和寬的 1/n （ n 為大於1的整數）。

注: 在步幅的使用中，目前多以Max pooling 來進行縮小尺寸


# 多輸入通道和多輸出通道

前面我們用到的輸入和輸出都是二維數組，但真實數據的維度經常更高。例如，彩色圖像在高和寬2個維度外還有RGB（紅、綠、藍）3個顏色通道

立體卷積是一個非常重要的概念，它不僅讓我們能夠處理彩色圖像，而且更重要的是，可以在一個單獨的層上使用多個濾波器。最重要的規則是，濾波器和你想在其上應用濾波器的圖像必須擁有相同的通道數。



![image](https://github.com/rockuass1235/deep-learning/blob/master/images/conv_multi_in.svg)

含2個輸入通道的互相關計算

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/conv_3d.jpg)


儘管我們這次從第三個維度讓矩陣中的數值對相乘。如果我們想在同一張圖像上應用多個濾波器，我們會為每個濾波器獨立地計算卷積，然後將計算結果逐個堆疊，最後將他們組合成一個整體。得到的張量（3D矩陣可以被稱作張量 Tensor）











# 原文出處

https://zhuanlan.zhihu.com/p/63220482

https://zhuanlan.zhihu.com/p/30994790

https://zh.gluon.ai


