
# Mxnet 安裝



# 前言


## 各種framework 支援模式

mxnet 同時支援命令式&聲明式，具有高度彈性的設計模式可以混合使用

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/framework.png)




# Mxnet isntall

Important: Make sure your installed CUDA version matches the CUDA version in the pip package. 

Check your CUDA version with the following command:

```
nvcc --version
```

You can either upgrade your CUDA install or install the MXNet package that supports your CUDA version.

```
pip3 install mxnet-cu90
```

# Install by MiniConda


```
mkdir d2l-zh && cd d2l-zh

curl https://zh.d2l.ai/d2l-zh-1.0.zip -o d2l-zh.zip

unzip d2l-zh.zip && rm d2l-zh.zip

conda env update -f environment.yml

```

啟動之前建立好的環境

```
# 若conda版本低于4.4，使用命令activate gluon
conda activate gluon  
jupyter notebook
```


# 使用GPU版的MXNet


第一步：卸載CPU 版本MXNet。如果你沒有安裝虛擬環境，可以跳過此步。否則假設你已經完成了安裝，那麼先激活運行環境，然後卸載CPU 版本的MXNet：


```
source activate gluon
pip  uninstall  mxnet

```

第二步：更新依賴為GPU版本的MXNet。使用文本編輯器打開之前文件夾下的文件environment.yml，將裡面的“mxnet”替換成對應的GPU版本。

例如，如果你電腦上裝的是8.0版本的CUDA，將該文件中的字符串“mxnet”改為“mxnet-cu80”。如果電腦上安裝了其他版本的CUDA（比如7.5、9.0、9.2等），對該文件中的字符串“mxnet”做類似修改

（比如改為“mxnet-cu75”、“mxnet-cu90”、“mxnet -cu92”等）。保存文件後退出。

第三步：更新虛擬環境。同前一樣執行

```
conda  env  update  -f  environment.yml
```



# MiniConda 遠端


```
pip install --upgrade pip
pip install jupyter
```






## 伺服器端設定

要遠端映射前，記得要先開啟putty登入帳號開啟jupyter notebook，然後不要關閉

```
jupyter notebook
```

記住以下這串TOKEN


![image](https://github.com/rockuass1235/deep-learning/blob/master/images/token.png)


## 用戶端設定

輸入以下指令:

要映射的port:8787; server開啟的port:8888; 帳號:godknow

Server ip:120.110.114.14

```
ssh -L8787:localhost:8888 godknow@120.110.114.14
```

用戶端端電腦上輸入上面指令，輸入完後CMD也不能關，關閉會導致port關閉

直接打開網頁輸入localhost:8787即可開啟映射到server的jupyter notebook


# 成功畫面

輸入剛剛記憶的TOKEN

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/success_remote.png)


















