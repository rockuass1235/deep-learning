




# 人臉偵測與人臉辨識

近年來隨著深度學習與顯示卡運算的技術突破，讓過往的傳統機器視覺有了天翻地覆的變化，傳統的LBP、HOG搭配SVM向量機的辨識率普遍上低於這些複雜的深度學習模型，

因此我們採用2015年由google提出來的Facenet(在LFW人臉資料庫以99.63%的最佳成績刷新了記錄)來進行人臉辨識。

整個人臉辨識分成兩個部分；影像中人臉的偵測與錨定和臉部特徵向量的擷取，最後與資料庫內的特徵資料利用Knn鄰近法則進行分類來取的辨識結果。


使用此方法的原因是:

*1.	因為一般來說，我們不太可能取得足夠量的訓練集資料來進行訓練，導致其他方法在人臉辨識上顯得不是這麼的好。

*2.	如果把每個人當作不同的類別去做訓練，有可能導致類別的數量遠大於資料維度而影響預測結果，而且每次添加新的人員時需要重新調整輸出維度。

*3.	傳統的CNN也許可以辨識出是人類但是對於更細節的辨識例如分辨是什麼人是有心而無力的。


# Dlib 簡介

dlib中提供的人臉檢測方法（使用HOG特徵或卷積神經網方法）

並使用提供的深度殘差網絡（ResNet）實現實時人臉識別，不過本文的目的不是構建深度殘差網絡，而是利用已經訓練好的模型進行實時人臉識別，實時性要求一秒鐘達到10幀以上的速率，並且保證不錯的精度。

opencv和dlib都是非常好用的計算機視覺庫，特別是dlib，前面文章提到了其內部封裝了一些比較新的深度學習方法，使用這些算法可以實現很多應用，比如人臉檢測、車輛檢測、目標追蹤、語義分割等等

``` Python

#安裝指令

pip install dlib


或以下GPU版本

#因為dlib是用C++語言編寫,編譯需要用到cmake
$ sudo apt install cmake
$ sudo apt install git
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build 
$ cd build
# 若cuda cuDNN 配置失敗，cmake不會啟動對cuda 的支援
$ cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
#以下為加速選項 
$ cmake .. -DUSE_SSE2_INSTRUCTIONS=1
$ cmake .. -DUSE_SSE4_INSTRUCTIONS=1
#將預設模式改release
$  cmake --build . --config Release 
$ cd ..
$ python setup.py install


```

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/dlib_setup.png)


# 實作流程

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/dlib_face_process.png)


# 資料處理


將照片資料大小按照比例縮小至N:R ，可以有效的減少資料量，加快處理速度 CV處理是按照長*寬 = array(col : row)

```Python

#將照片格式按比例縮小至(n, ?) 
def fix(img):
    n  = 400
    r = n/img.shape[1]    #col 
    dim = (n, int(r*img.shape[0]))  #pic = (col , row)
    img = cv2.resize(img, dim)
    return img

```



# 臉部區域偵測

臉部區域偵測會將圖片裡所有的臉部區域找出來回傳出一個list

如果找不到則回傳NONE

為了減少資訊量建議用灰階圖片代替


```Python

def face_area(img):
    
    detector = dlib.get_frontal_face_detector()  # face detector
    
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    
    if len(dets) == 0 or len(dets) > 1:
        return None
    
    return dets[0]


```

# 臉部回歸68點特徵位置

透過前面取的臉部區域輸入另一個CNN網路取得臉部對應68個點座標

``` Python

def get_68p_facemark(img):
    
    p = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    det = face_area(img)
    
    if det == None:
        return None
    
    return p(img, det)


```

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/dlib_facemark.png)


# 輸出量化特徵值

局部圖像特徵是計算機視覺中常常討論的議題之一，在尋找圖像中的對應向量以及物體特徵描述中有著重要的作用。

其核心想法是要有一個可靠的圖像對應向量的關係集合，也就是說在不同的角度或場合，對於相同的物體都能得到相同的特徵向量。

而能夠精準建立圖像之間點與點之間可靠的對應關係的演算法，都需要仰賴設計出一個完美的最佳化局部圖像特徵運算子，讓物體識別可以泛化應用在很多場景。

```Python

def get_128d_features(img):
    
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    facemark = get_68p_facemark(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    
    if facemark is None:
        return None
    
    features = facerec.compute_face_descriptor(img, facemark) # 3.描述子提取，128D向量
    
    return np.array(features).reshape(128)


```


![image](https://github.com/rockuass1235/deep-learning/blob/master/images/dlib_face_features.png)


# K 最近鄰法則分類

利用歐式距離進行相關度分類，閥值控制距離至少小於多少(true-positive)，如果不滿足則輸出?????

```Python

def nearestClass(face_descriptor,data, labels, threshold):
    
#  distance is caculate by euclidean_distance

    t =  face_descriptor - data      #face_feature 與 所有data的距離
    e = np.linalg.norm(t, axis = 1, keepdims = True)  #linalg=linear（線性）+algebra（代數），norm則表示範數。
    min_d = e.min()    #value asign to min_d
    
    #==============debug==================
    print('distance: ', min_d)
    #=====================================
    
    
    if min_d > threshold:
        return '??????'
    

    index = np.argmin(e)  #  min_index asign to index
    return labels[index]

```




# 結果

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/dlib_result1.png)

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/dlib_result2.png)







