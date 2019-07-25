

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

