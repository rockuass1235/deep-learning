


# 前言

## Siamese network 孿生神經網絡--一個簡單神奇的結構

十九世紀泰國出生了一對連體嬰兒，當時的醫學技術無法使兩人分離出來，於是兩人頑強地生活了一生，1829年被英國商人發現，進入馬戲團，在全世界各地表演，

1839年他們訪問美國北卡羅萊那州後來成為“ 玲玲馬戲團 ”的台柱，最後成為美國公民。1843年4月13日跟英國一對姐妹結婚，恩生了10個小孩，昌生了12個，姐妹吵架時，兄弟就要輪流到每個老婆家住三天。

1874年恩因肺病去世，另一位不久也去世，兩人均於63歲離開人間。兩人的肝至今仍保存在費城的馬特博物館內。從此之後“ 暹羅雙胞胎 ”（Siamese twins）就成了連體人的代名詞，也因為這對雙胞胎讓全世界都重視到這項特殊疾病。


![](https://github.com/rockuass1235/deep-learning/blob/master/images/twins.jpg)


簡單來說，Siamese network就是“連體的神經網絡”，神經網絡的“ 連體”是通過共享權值來實現的，如下圖所示。

在代碼實現的時候，甚至可以是同一個網絡，不用實現另外一個，因為權值都一樣。對於siamese network，兩邊可以是lstm或者cnn，都可以。


![](https://github.com/rockuass1235/deep-learning/blob/master/images/twins_network.jpg)


## 孿生神經網絡的用途是什麼？

簡單來說，衡量兩個輸入的相似程度。孿生神經網絡有兩個輸入（Input1 and Input2）,將兩個輸入feed進入兩個神經網絡（Network1 and Network2），這兩個神經網絡分別將輸入映射到新的空間，形成輸入在新的空間中的表示。

**通過Loss的計算，評價兩個輸入的相似度** 。

EX: 簽名筆跡驗證、 人臉辨識

先上結論：孿生神經網絡用於處理兩個輸入"比較類似"的情況。偽孿生神經網絡適用於處理兩個輸入"有一定差別"的情況。比如，我們要計算兩個句子或者詞彙的語義相似度，使用siamese network比較適合；

如果驗證標題與正文的描述是否一致（標題和正文長度差別很大），或者文字是否描述了一幅圖片（一個是圖片，一個是文字），就應該使用pseudo-siamese network。

也就是說，要根據具體的應用，判斷應該使用哪一種結構，哪一種Loss。





# Triplet Loss

Triplet loss最初是在FaceNet: A Unified Embedding for Face Recognition and Clustering論文中提出的，可以學到較好的人臉的embedding

輸入是一個三元組 <a, p, n>

* a： anchor
* p： positive,與a是同一類別的樣本
* n： negative,與a是不同類別的樣本


![](https://github.com/rockuass1235/deep-learning/blob/master/images/triplet_loss.jpg)



# 資料來源


https://zhuanlan.zhihu.com/p/35040994

http://lawlite.me/2018/10/16/Triplet-Loss%E5%8E%9F%E7%90%86%E5%8F%8A%E5%85%B6%E5%AE%9E%E7%8E%B0/