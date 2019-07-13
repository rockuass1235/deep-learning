# 深度學習的activation function


**優先使用ReLU (Rectified Linear Unit) 函數作為神經元的activation function**

## 理論

深度學習的基本原理是基於人工神經網絡，信號從一個神經元進入，經過**非線性的** activation function，傳入到下一層神經元；
再經過該層神經元的activate，繼續往下傳遞，如此循環往復，直到輸出層。正是由於這些非線性函數的反复疊加，才使得神經網絡有足夠的capacity來抓取複雜的pattern，在各個領域取得state-of-the-art的結果。
顯而易見，activation function在深度學習中舉足輕重，也是很活躍的研究領域之一。
目前來講，<font color=#DC143C>**選擇怎樣的activation function不在於它能否模擬真正的神經元，而在於能否便於優化整個深度神經網絡。**</font>
下面我們簡單聊一下各類函數的特點以及為什麼現在優先推薦ReLU函數。
