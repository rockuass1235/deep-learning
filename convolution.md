# Convolution Layer

卷積神經網絡（convolutional neural network）是含有捲積層（convolutional layer）的神經網絡。最常見的二維卷積層。
它有高和寬兩個空間維度，常用來處理圖像數據。我們將介紹簡單形式的二維卷積層的工作原理。

在二維卷積層中，一個二維輸入數組和一個二維核（kernel）數組通過互相關運算輸出一個二維數組。我們用一個具體例子:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/conv.gif)

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/conv_formula.jpg)

它的物理意義大概可以理解為：系統某一時刻的輸出是由多個輸入共同作用（疊加）的結果。

卷積核上所有作用點依次作用於原始像素點後（即乘起來），線性疊加的輸出結果，即是最終卷積的輸出，也是我們想要的結果，我們稱為destination pixel.

