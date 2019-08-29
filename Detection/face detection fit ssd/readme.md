



## 前言

手機的人臉辨識是目前大家對於類神經網路最直接接觸的領域，傳統上我們將人臉辨識分成 **人臉偵測與人臉識別** 兩個部分。

而這裡我們用SSD來進行影像中人臉的偵測。


## 相關需要labrary

```

pip install mxnet
pip install numpy

```




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




#### 特徵抽取模組

CNN網絡一般前面的特徵圖比較大，後面會逐漸採用stride=2的捲積或者pool來降低特徵圖大小，通道數翻倍避免特徵圖縮小後損失部分資訊

這樣的意義在於不同的捲積層提供不同的特徵抽取能力，寬、高個減半後不但減少資料大小，而且還提供更寬廣的視野大小

```Python

def blk(num, channels):

    net = nn.Sequential()
    with net.name_scope():
        for _ in range(num):
            net.add(nn.Conv2D(channels, kernel_size=3, padding=1))
            net.add(nn.BatchNorm(in_channels=channels))
            net.add(nn.Activation('relu'))
            
            
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
    return net

```


#### 採用多尺度特徵圖用於檢測


我們對於抽取特徵次數較少的特徵圖產生大小比例較小的anchor box，抽取次數較多的特徵圖生成大小比例較大的特徵圖。透過不同尺度大小特徵圖產稱不同大小的錨框滿足對應真實框的需求。


![](https://github.com/rockuass1235/deep-learning/blob/master/images/object-detection_0.svg)

![](https://github.com/rockuass1235/deep-learning/blob/master/images/object-detection_1.svg)

![](https://github.com/rockuass1235/deep-learning/blob/master/images/object-detection_2.svg)






#### Anchor Box


在Yolo中，每個單元預測多個邊界框，但是其都是相對這個單元本身（正方塊），但是真實目標的形狀是多變的，Yolo需要在訓練過程中直接預測真實框大小。

而SSD借鑒了Faster R-CNN中anchor的理念，每個單元設置尺度或者長寬比不同的先驗框，預測的邊界框（bounding boxes）是以這些先驗框為基準的，在訓練過程中預測先驗框要經過何種修正更逼近真實框，在一定程度上減少訓練難度。

注意:

SSD的與YOLO 相同會對每個預測生成的bbox產生一個confidence，但定義與YOLO不同(YOLO的cofidence = (1-背景機率)x(IOU))， SSD confidence其值 = 最高類別機率值

SSD將背景也當做了一個特殊的類別，如果檢測目標共有N個類別，SSD其實需要預測N+1個類別機率





#### 實作



SSD MODEL與 Faster rcnn有些類似， 透過數層的捲積層組成一個抽取圖片特徵的模組，在抽取後得到的特徵圖上依照各個像素位置產生數個錨框並利用特徵圖生成對錨框類別與錨框offset的預測。

之後將得到的特徵圖記趣經過下一個模組抽取特徵產生錨框重複上述步驟， 最後 **輸出全部產生的錨框、類別預測、錨框offset**


我們先設計一個特徵抽取模組並結合輸出 錨框、類別、offset

```Python

def cls_blk(anchors_num, cls_num):
    
    out = anchors_num * (cls_num+1)
    
    return nn.Conv2D(out, kernel_size=3, padding = 1)

def bbox_blk(anchors_num):
    
    out = anchors_num * 4
    
    return nn.Conv2D(out, kernel_size=3, padding = 1)





class MyBlk(nn.Block):
    
    def __init__(self, blk, cls_num, size, ratio, **kwargs):
        super(MyBlk, self).__init__(**kwargs)
        
        
        self.size = size
        self.ratio = ratio
        N = len(size) + len(ratio) - 1
        
        self.blk = blk
        self.cls_blk = cls_blk(N, cls_num)
        self.bbox_blk = bbox_blk(N)
        
    def forward(self, x):
        
        
        yhat = self.blk(x)
        anchors = contrib.nd.MultiBoxPrior(yhat, sizes = self.size, ratios = self.ratio)
        cls_yhat = self.cls_blk(yhat)
        bbox_yhat = self.bbox_blk(yhat)
        
        return yhat, anchors, cls_yhat, bbox_yhat

```

將數個這樣的模組串接再一起，並輸出每個階段產生的anchor box、 cls predict、 bbox offset


```Python




class MyMod(nn.Block):

    def __init__(self, classes, ctx=mx.cpu(), **kwargs):
        super(MyMod, self).__init__(**kwargs)
        self.classes = classes
        self.ctx = ctx

        net = gluon.model_zoo.vision.resnet18_v2(pretrained=True).features

        self.net_0 = MyBlk(net[:5], classes, [0.8 / (2) ** 2.5], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_1 = MyBlk(net[5:6], classes, [0.8 / (2) ** 2], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_2 = MyBlk(net[6:7], classes, [0.8 / (2) ** 1.5], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_3 = MyBlk(net[7:8], classes, [0.8 / (2) ** 1], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_4 = MyBlk(net[8:9], classes, [0.8 / (2) ** 0.5], [0.5, 1, 2, 1.618, 1 / 1.618])
        self.net_5 = MyBlk(net[9:12], classes, [0.8], [0.5, 1, 2, 1.618, 1 / 1.618])



	# extend mxnet.gluon.nn.Block 需要 implement forward函數
	
    def forward(self, x):
        anchors, cls_yhats, bbox_yhats = [], [], []

        for i in range(6):
            net = getattr(self, 'net_%d' % i)
            x, anch, cls_yhat, bbox_yhat = net(x)

            cls_yhat = cls_yhat.transpose((0, 2, 3, 1)).flatten()
            cls_yhat = cls_yhat.reshape((cls_yhat.shape[0], -1, self.classes))

            bbox_yhat = bbox_yhat.transpose((0, 2, 3, 1)).flatten()
            bbox_yhat = bbox_yhat.reshape((bbox_yhat.shape[0], -1, 4))

            anchors.append(anch)
            cls_yhats.append(cls_yhat)
            bbox_yhats.append(bbox_yhat)

        return nd.concat(*anchors, dim=1), nd.concat(*cls_yhats, dim=1), nd.concat(*bbox_yhats, dim=1)


	# 以下是方便predict使用的函數嵌入模型中
    def get_bboxes(self, X):

        anchors, cls_yhat, bbox_yhat = self(X)
        cls_yhat = cls_yhat.softmax().transpose((0, 2, 1))
        out = contrib.nd.MultiBoxDetection(cls_yhat, bbox_yhat.flatten(), anchors)
        bboxes = []

        for i, img in enumerate(out):
            idx = img[:, 0] > -1
            idx = np.where(idx.asnumpy() >= 0.5)
            bboxes.append(img[idx].as_in_context(mx.cpu()))

        return bboxes

    def reset_ctx(self, ctx):

        self.ctx = ctx
        self.collect_params().reset_ctx(self.ctx)

    def predict(self, x, threshold=0.5):

        x = x.as_in_context(self.ctx)
        bboxes = self.get_bboxes(x)
        Y = nd.zeros(shape=(len(bboxes), 6))

        for i, gt in enumerate(bboxes):

            idx = gt[:, 1] >= threshold
            idx = np.where(idx.asnumpy() >= 0.5)[0]
            if len(idx) <= 0:
                continue

            idx = idx[0]
            Y[i] = gt[idx]
        return Y
```




## Loss function 設計


類別的loss function使用 softmax cross entropy

offset的loss function 使用 L1Loss (offset有方向性)


```Python

from mxnet.gluon import loss as gloss

def get_label(anchors, y, yhat):
    
    bbox_y, bbox_masks, cls_y = contrib.nd.MultiBoxTarget(anchors, y, yhat.transpose((0, 2, 1)))
    return bbox_y, bbox_masks, cls_y



cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()

def cost(cls_yhat, cls_y, bbox_yhat, bbox_y, bbox_masks):
    
    bbox_yhat = bbox_yhat.flatten()
    return cls_loss(cls_yhat, cls_y) + bbox_loss(bbox_yhat*bbox_masks, bbox_y * bbox_masks)


def predict(net, x):
    
    
    anchors, cls_yhat, bbox_yhat = net(x.as_in_context(ctx))
    
    cls_yhat = cls_yhat.softmax()
    cls_yhat = cls_yhat.transpose((0, 2, 1))  # shape = batch_size, cls_one_hot, num_anchors
    bbox_yhat = bbox_yhat.flatten()
    
    return contrib.nd.MultiBoxDetection(cls_yhat, bbox_yhat, anchors, nms_threshold= 0.5)


```




## 結果


```Python

epoch: 40, train loss: 0.014245, time: 17.651 sec
epoch: 41, train loss: 0.014170, time: 17.789 sec
epoch: 42, train loss: 0.014145, time: 17.644 sec
epoch: 43, train loss: 0.013985, time: 17.474 sec
epoch: 44, train loss: 0.014005, time: 17.515 sec
epoch: 45, train loss: 0.013980, time: 17.424 sec
epoch: 46, train loss: 0.013804, time: 17.593 sec
epoch: 47, train loss: 0.013783, time: 17.620 sec
epoch: 48, train loss: 0.013742, time: 17.631 sec
epoch: 49, train loss: 0.013635, time: 17.782 sec
epoch: 50, train loss: 0.013655, time: 17.584 sec

```

![](https://github.com/rockuass1235/deep-learning/blob/master/images/face_detect.png)
`















## 資料來源

http://www.cvmart.net/community/article/detail/148

http://zh.gluon.ai/chapter_computer-vision/ssd.html

https://zhuanlan.zhihu.com/p/33544892





