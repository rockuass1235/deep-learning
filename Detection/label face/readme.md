# label face

本檔案使用 dlib 提供的人臉偵測器進行自動標籤人臉位置

預設每張圖標記20個真實框 不足部分類別補-1， 超過自動忽略。 如果要增加真實框數量可調整label.py 內的 gt_MAX = ?

# 使用方式

```

python label.py (圖檔資料夾位置) (檔案名稱 ex: test) 0.8(分割比例)

```
執行後分別在目錄產生 test_train.lst、test_train.rec、test_test.lst、test_test.rec 四個檔案

# Data argumentation

為了訓練SSD對於小型真實框matching能力， 可藉由輸入:

```
python label.py (圖檔資料夾位置) (label.py 產生的lst檔案名稱 ex: test.lst) (label.py 產生的rec檔案名稱 ex: test.rec)
```

執行後分別在目錄產生 test_aug.lst、test_aug.rec 2個檔案，並在圖檔目錄下增加augs資料夾存放資料增強後產生的圖檔


# test


```
python test.py (圖檔資料夾位置) (label.py 產生的lst檔案名稱 ex: test.lst) (label.py 產生的rec檔案名稱 ex: test.rec)

```

執行後自動 plt.show()顯示圖片標籤情況

