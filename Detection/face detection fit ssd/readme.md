



## 前言

手機的人臉辨識是目前大家對於類神經網路最直接接觸的領域，傳統上我們將人臉辨識分成 **人臉偵測與人臉識別** 兩個部分。

而這裡我們用SSD來進行影像中人臉的偵測。





## 數據集製作

我們選用了lfw_5590數據集作為我們的數據集， 5540筆資料作為訓練集; 後面50筆作為測試集。

這裡用dlib提供的 dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat') 來對資料生成 ground truth 標籤

ground truth 的標籤格式如下:  [ground_truth_num, class, x1, y1, x2, y2] 

將所有的ground truth 結合在一起後得到標籤集， 格式如下:[ batch, ground_truth_num, cls, x1, y1, x2, y2]

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





