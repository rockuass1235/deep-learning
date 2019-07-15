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

所以我們得到以下公式: ![image](https://github.com/rockuass1235/deep-learning/blob/master/images/padding_formula.jpg)

