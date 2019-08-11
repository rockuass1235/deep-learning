

# 錨框：Anchor box

# 前言

傳統的detection主流方法是DPM(Deformable parts models)，在VOC2007上能到43%的mAP，雖然DPM和CNN看起來差別很大，但RBG大神說“Deformable Part Models are Convolutional Neural Networks”（http:/ / arxiv.org/abs/1409.5403）。

CNN流行之後，Szegedy做過將detection問題作為回歸問題的嘗試（Deep Neural Networks for Object Detection），但是效果差強人意，在VOC2007上mAP只有30.5%。 **既然回歸方法效果不好，而CNN在分類問題上效果很好，那麼為什麼不把detection問題轉化為分類問題呢？**

所以目標檢測的思想是，首先在圖片中尋找 **可能存在物體的位置（regions），然後再判斷這個位置裡面的物體是什麼東西** 





物件偵測有以下幾種方法:

* 滑動窗口(Sliding window)

這是比較原始的目標檢測方法，給定一個固定尺寸的窗口，根據設定的步伐，一步一步的從左至右、從上至下滑動，把每個窗口輸入到卷積神經網絡中進行預測和分類，這樣做有一個非常致命的缺點:

由於物體的大小是不可預知的，所以還要用不同大小的框框去偵測。但是 Sliding Window 是非常暴力的作法，對單一影像我們需要掃描非常多次，每掃一次都需要算一次 CNN，這將會耗費大量的運算資源，而且速度慢，根本無法拿來應用！



* 建議區域(region proposal)

與其用 Sliding Window 的方式掃過一輪，R-CNN 的作法是預先篩選出約 N 個可能的區域，這是R-CNN系列中核心的思想。

RCNN算法分為4個步驟 

* 候選區域生成： 一張圖像生成1K~2K個候選區域（採用Selective Search 方法）
* 特徵提取： 對每個候選區域，使用深度卷積網絡提取特徵（CNN） 
* 類別判斷： 特徵送入每一類的SVM 分類器，判別是否屬於該類 
* 位置精修： 使用回歸器精細修正候選框位置 



![image](https://github.com/rockuass1235/deep-learning/blob/master/images/rcnn.png)


#### Selective Search


R-CNN 用來篩選 Region Proposals 的方法稱之為 Selective Search ，而 Selective Search 又是基於 Felzenszwal 於 2004 年發表的論文 Graph Base Image Segmentation。

圖像經由 Graph Base Image Segmentation 可以切出數個 Segment 來，如下圖：


![image](https://github.com/rockuass1235/deep-learning/blob/master/images/selective_search.png)


#### Faster RCNN算法

Faster R-CNN中，模型中使用了兩個神經網絡，一個是是CNN(用於分類)，一個是RPN(Regional Proposal Network)

RPN 用於取的作用是代替以往rcnn使用的selective search的方法尋找圖片裡面可能存在物體的區域。 當一張圖片輸入resnet或者vgg，在最後一層的feature map上面，尋找可能出現物體的位置。

這時候分別以這張feature map的每一個點為中心，在原圖上畫出9個尺寸不一anchor。然後計算anchor與GT（ground truth） box的iou（重疊率），滿足一定iou條件的anchor，便認為是這個anchor包含了某個物體。

**所以region proposal就參與了判斷物體可能存在位置的過程。讓模型學會去看哪裡有物體，GT box就是給它進行參考，了解與真實邊框的差異程度。**







* 錨框(anchor box)


anchor boxes是學習卷積神經網絡用於目標識別過程中最重要且最難理解的一個概念。這個概念最初是在Faster R-CNN中提出，此後在SSD、YOLOv2、YOLOv3等優秀的目標識別模型中得到了廣泛的應用

每個像素作為anchor box的中心點稱之為 anchor point，然後將事先準備好的 k 個不同尺寸比例的 box 以同一個 anchor point 去計算可能包含物體的機率(score)，取機率最高的 box。這 k 個 box 稱之為 anchor box。

在經過一系列卷積和池化之後，在feature map層使用anchor box，如上圖所示，經過一系列的特徵提取，最後針對3x3的網格會得到一個3x3xKx8的特徵層，其中K是anchor box的個數

8代表每個anchor box包含的變量數，分別是4個位置偏移量、3個類別(one-hot標註方式)、1個anchor box標註(如果anchor box與真實邊框的交並比最大則為1，否則為0)。
































# Intersection Over Union(IOU, 交並比)

剛剛提到某個錨框較好地覆蓋了圖像中的狗。如果該目標的真實邊界框已知，這裡的“較好”該如何量化呢？

一種直觀的方法是衡量錨框和真實邊界框之間的相似度。我們知道，Jaccard係數（Jaccard index）可以衡量兩個集合的相似度。給定集合 A 和 B ，它們的Jaccard係數即二者交集大小除以二者並集大小：

## J(A,B)=|A∩B|/|A∪B|

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/iou.svg)

我們將使用交並比來衡量錨框與真實邊界框以及錨框與錨框之間的相似度。










# NMS—非極大值抑制演算法

## 前言

在目標檢測中，得到多個候選框及其置信度得分。非極大值抑制演算法(NMS)對多個候選框，去除重合率大的冗餘候選框，得到最具代表性的結果，以加快目標檢測的效率。

## 作法

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/nms_original.jpg)


* 將所有框的得分排序，選中最高分及其對應的框

假設圖中有A：0.75、B:0.98 、C：0.83、D:0.67、E：0.81，將置信度升序排序為D:0.67、A：0.75、E：0.81、C：0.83、B:0.98，選中得分最高的B:0.98。

* 遍歷其餘的框，如果和當前最高分框的重疊面積(IOU)大於一定閾值，我們就將框刪除。

由B:0.98對其餘A、C、D、E框計算IOU，B與A、C的IOU>閾值，刪除A、C框。第一輪得到B：0.98、D:0.67、E：0.81。

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/nms_round1.jpg)

從未處理的D、E框中繼續選一個得分最高的，重複上述過程。最後保留B、E框

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/nms_round2.jpg)







# 原文出處


https://www.itread01.com/content/1541722582.html

https://zhuanlan.zhihu.com/p/63024247

https://www.zhihu.com/question/35887527

https://www.zhihu.com/question/265345106

https://medium.com/@syshen/%E7%89%A9%E9%AB%94%E5%81%B5%E6%B8%AC-object-detection-740096ec4540

http://zh.gluon.ai/chapter_computer-vision/anchor.html





