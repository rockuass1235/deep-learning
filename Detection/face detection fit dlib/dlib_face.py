#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import dlib
import numpy as np
import cv2




def face_area(img):
    
    detector = dlib.get_frontal_face_detector()  # face detector
    
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    
    if len(dets) == 0 or len(dets) > 1:
        return None
    
    return dets[0]
    
def get_68p_facemark(img):
    
    p = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    det = face_area(img)
    
    if det == None:
        return None
    
    return p(img, det)


def get_128d_features(img):
    
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    facemark = get_68p_facemark(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    
    if facemark is None:
        return None
    
    features = facerec.compute_face_descriptor(img, facemark) # 3.描述子提取，128D向量
    
    return np.array(features).reshape(128)



#將照片格式按比例縮小至(n, ?) 
def fix(img):
    n  = 400
    r = n/img.shape[1]    #col 
    dim = (n, int(r*img.shape[0]))  #pic = (col , row)
    img = cv2.resize(img, dim)
    return img


def pos(ret, fix = 0):
  
    pos_start = (ret.left() - fix, ret.top() - fix)
    pos_end = (ret.right() + fix, ret.bottom() + fix)
    
    return pos_start, pos_end
 
    
def show_data(frame, label):
    
    #labels = get_labels(labels)
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(label)
    plt.show()
    
    


# In[27]:



def show_details(frame, label):
    
    #圖片大小處理
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  #顏色排列轉換
    img = fix(img) 
  
    
    #計算矩形範圍
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #gray-scale
    det = face_area(img_gray)
    if det is not None:
        face = img[det.top():det.bottom(), det.left():det.right(), :].copy()
        #  劃出矩形範圍
        p1, p2 = pos(det, 20)
        cv2.rectangle(img, p1, p2, (255,0,0), 8)
        #  標出臉上68個點座標
        p = get_68p_facemark(img_gray)
        for i in range(68):
            x, y = p.part(i).x, p.part(i).y
            cv2.circle(img, (x,y), 1, (0, 255, 0), 1)
    else:
        face = None
    
    
        
   #顯示圖片 
    _, figs = plt.subplots(1,2, figsize = (10, 10))
    
    figs[0].imshow(img)
    figs[0].set_title(label)
    figs[0].axes.get_xaxis().set_visible(False)
    figs[0].axes.get_yaxis().set_visible(False)
    if face is not None:
        figs[1].imshow(face)
    figs[1].set_title(label)
    figs[1].axes.get_xaxis().set_visible(False)
    figs[1].axes.get_yaxis().set_visible(False)
    plt.show()
    
        


# In[20]:


#  build database
import os
import json




def build_data(video, labelName):
        
    
    data = []                  #定義空的data存放人臉的特徵
    labels = []                #定義空的list存放人臉的標籤

    i = 0
    while video.isOpened():  # 判斷攝像頭是否啟動
        flag, frame = video.read()  # 讀取一幀數據

        if not flag:  # 讀取失敗
            print('failed')
            break
        os.system('cls')
        i+=1
        print(i)
        #如果圖太大的話需要壓縮
        img = fix(frame)
        print(img.shape)
        feature = get_128d_features(img)
        show_details(img, labelName)  #  顯示資料
        
        
        #資料保存
        if feature is not None:
            data.append(feature)
            labels.append(labelName)
        
        
       
       

    return np.array(data), labels


def load_data():
    labelFile=open('label.txt','r')
    label = json.load(labelFile)  # 載入本地人臉庫的標籤
    labelFile.close()
    data = np.loadtxt('faceData.txt',dtype=float) 
    return data, label

def save_data(data, labels):

    #保存人臉特徵合成的矩陣到本地
    #使用json保存list到本地
    Data_file = 'faceData.txt'
    label_file = 'label.txt'
    
    
    np.savetxt(Data_file, data, fmt='%f')                                                          
    file = open(label_file,'w')                                      
    json.dump(labels, file)                                                                         
    file.close()
    
def add_data(data, labels):
    
    x, y = load_data()
    x = np.append(x, data, axis = 0)
    y = y+labels
    save_data(x, y)
    


# In[11]:



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



# In[28]:



img = cv2.imread('yee_2.jpg')
show_details(img, 'yee')
feature = get_128d_features(img)
print(feature)


# In[34]:


Data_file = 'faceData.txt'
label_file = 'label.txt'

video = cv2.VideoCapture('410528478.avi')
data, labels = build_data(video, 'kalas')


# In[35]:


add_data(data, labels)


# In[44]:


x, y = load_data()
img = cv2.imread('test.jpg')
true_label = 'godknow'
feature = get_128d_features(img)

print('this is %s ' % nearestClass(feature, x, y, 0.4))
show_details(img, true_label)


# In[ ]:




