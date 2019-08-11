



## 前言

手機的人臉辨識是目前大家對於類神經網路最直接接觸的領域，傳統上我們將人臉辨識分成 **人臉偵測與人臉識別** 兩個部分。

而這裡我們用SSD來進行影像中人臉的偵測。





## 數據集製作

我們選用了lfw_5590數據集作為我們的數據集， 5540筆資料作為訓練集; 後面50筆作為測試集。

這裡用dlib提供的 dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat') 來對資料生成 ground truth 標籤

ground truth 的標籤格式如下:  [ground_truth_num, class, x1, y1, x2, y2] 

將所有的ground truth 結合在一起後得到標籤集， 格式如下:[ batch, ground_truth_num, cls, x1/w, y1/h, x2/w, y2/h]

產生的數據集為 gluon.NDarray 格式， 檔案名稱為 face_data.pkl




#### 問題:

每個圖像內的ground truth 數量都不同，不但結合困難，於且無法使用batch size 進行訓練(batch training 需資料格式一致)

每張圖片中不一定人臉數量都一樣，也就是ground truth 數量不同


#### Solution:

我們透過添加 [0, 0, 0, 0, 0] 將ground truth 的數量保持一致， 在用contrib.MultiBoxTarget() 對每個anchor box 產生對應 labels 時，會自動忽略全0項。

```Python

def get_data(path):
    
    dir = read_img(path)
    X = nd.zeros(shape = (len(dir), 250, 250, 3), dtype = 'uint8')
    for x, d in zip(X, dir):
        img = image.imread(d)
        x[:] = img
        
        
         
    # 資料的ground truth數量不一同無法使用batch size trainning，
    # 可補[0,0,0,0,0] 將其大小擴展為一致，MultiboxTarget計算時會自動忽略不影響輸出
    Y = nd.zeros(shape = (len(dir), 10, 5))
    
    
    
    
    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    
    for i, x in enumerate(X):
        dets = detector(x.asnumpy(), 1)
        gt = nd.zeros(shape = (10, 5))
       
        for j, det in enumerate(dets):
            x1, y1, x2, y2 = det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()
            gt[j] = nd.array([0, x1, y1, x2, y2])
        
        Y[i] = gt/250
            
        
    return X, Y
```



## SSD MODEL


目標檢測近年來已經取得了很重要的進展，主流的算法主要分為兩個類型（參考RefineDet）：

* （1）two-stage方法，如R-CNN系算法，其主要思路是先通過啟發式方法（selective search）或者CNN網絡（RPN)產生一系列稀疏的候選框，然後對這些候選框進行分類與回歸，two-stage方法的優勢是準確度高；

* （2）one-stage方法，如Yolo和SSD，其主要思路是均勻地在圖片的不同位置進行密集抽樣，抽樣時可以採用不同尺度和長寬比，然後利用CNN提取特徵後直接進行分類與回歸，整個過程只需要一步，所以其優勢是速度快，但是均勻的密集採樣的一個重要缺點是訓練比較困難

SSD算法在準確度和速度（除了SSD512）上都比Yolo要好很多。圖2給出了不同算法的基本框架圖，對於Faster R-CNN，其先通過CNN得到候選框，然後再進行分類與回歸，而Yolo與SSD可以一步到位完成檢測。

相比Yolo，SSD採用CNN來直接進行檢測，而不是像Yolo那樣在全連接層之後做檢測。其實採用卷積直接做檢測只是SSD相比Yolo的其中一個不同點，另外還有兩個重要的改變，一是SSD提取了不同尺度的特徵圖來做檢測，

大尺度特徵圖（較靠前的特徵圖）可以用來檢測小物體，而小尺度特徵圖（較靠後的特徵圖）用來檢測大物體；二是SSD採用了不同尺度和長寬比的先驗框（Prior boxes, Default boxes ，在Faster R-CNN中叫做錨，Anchors）。

Yolo算法缺點是難以檢測小目標，而且定位不准，但是這幾點重要改進使得SSD在一定程度上克服這些缺點。



#### 採用多尺度特徵圖用於檢測

CNN網絡一般前面的特徵圖比較大，後面會逐漸採用stride=2的捲積或者pool來降低特徵圖大小，通道數翻倍避免特徵圖縮小後損失部分資訊

我們對於抽取特徵次數較少的特徵圖產生 大小比例較小的anchor box，抽取次數較多的特徵圖生成大小比例較大的特徵圖。透過不同尺度大小特徵圖產稱不同大小的錨框滿足對應真實框的需求。


![](https://github.com/rockuass1235/deep-learning/blob/master/images/object-detection_0.svg)

![](https://github.com/rockuass1235/deep-learning/blob/master/images/object-detection_1.svg)

![](https://github.com/rockuass1235/deep-learning/blob/master/images/object-detection_2.svg)
















SSD MODEL與 Faster rcnn有些類似， 透過數層的捲積層組成一個抽取圖片特徵的模組，在抽取後得到的特徵圖上依照各個像素位置產生數個錨框並利用特徵圖生成對錨框類別與錨框offset的預測。

之後將得到的特徵圖記趣經過下一個模組抽取特徵產生錨框重複上述步驟， 最後 **輸出全部產生的錨框、類別預測、錨框offset**
















## 資料來源

http://www.cvmart.net/community/article/detail/148

http://zh.gluon.ai/chapter_computer-vision/ssd.html





