

# 各種framework 支援模式

mxnet 同時支援命令式&聲明式，具有高度彈性的設計模式可以混合使用

# GPU Computing配置 (Ubuntu 18.04)


## GPU driver 安裝


檢查顯示卡資訊

``` bash
#check device information
ubuntu-drivers devices
```

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/gpu_device.png)

2選1安裝 driver

```bash


#install driver automatically
sudo ubuntu-drivers autoinstall


#or self install
sudo apt install nvidia-390

```

安裝完後重開機

```
#reboot your system
sudo reboot
```


# 安裝cuda9.2

本安裝cuda9.2 為dlib & mxnet 配置環境需求

tensorflow 官方目前20180822 所支援最新CUDA僅到9.0


有點類似於word一樣（高版本word能打開低版本的word文件.）18.04版本的系統，能夠安裝16.04版本對應的CUDA

根據cuDNN的版本，目前，較為完善的，是cuDNN v7.0.5 ,其適用於CUDA 9.1版本

但最新版的mxnet 為mxnet-cuda92 故安裝cuda9.2

Cuda web page: https://developer.nvidia.com/cuda-toolkit-archive



## CUDA Environment setting

開始安裝：

```bash

#downgrade gcc
#cuda only support  gcc/g++ lower than 6.0 version(default: 7.3)

sudo apt-get install gcc-4.8 
sudo apt-get install g++-4.8


# Change the original link from gcc – 7.3 to gcc-4.8
cd /usr/bin
sudo mv gcc gcc.bak #backup
sudo ln -s gcc-4.8 gcc #relocate
sudo mv g++ g++.bak 
sudo ln -s g++-4.8 g++

```

#### Options


![image](https://github.com/rockuass1235/deep-learning/blob/master/images/cuda_opt.png)

對應的有一下2個文件，需要統統下載，第一個是主文件，後1個相當於補丁

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/cuda_patch.png)


## Install cuda

```bash

#輸入命令安裝Base Installer：
sudo sh cuda_9.1.85_387.26_linux.run


# install 3 patch
sudo sh cuda_9.1.85.1_linux.run 
sudo sh cuda_9.1.85.2_linux.run 
sudo sh cuda_9.1.85.3_linux.run


```

安裝CUDA TOOLKIT 選項問題:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/cuda_toolkit_opt.png)


出現summary 代表安裝成功

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/cuda_summary.png)


## 設定環境變數:

```bash

sudo nano ~/.bashrc


#Notice: 根據自己的版本，修改cuda-9.2/9.0... 
# add the following below to the end of file
#cuda setting
export PATH=/usr/local/cuda-9.2/bin${PATH:+:$PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


#reboot
source ~/.bashrc

```

## 當程序跑完後CUDA9.2 就安裝完成了



# Cudnn 安裝


安裝cuDNN（針對cuda 9.2）

cuDNN 的安裝，就是將cuDNN 包內的文件，拷貝到cuda文件夾中即可。

根據cuDNN的版本，目前，較為完善的，是cuDNN v7.0.5 ,其適用於CUDA 9.2版本

cuDNN web page: https://developer.nvidia.com/cudnn


```bash

#複製cuDNN內容到cuda相關文件夾內
#注意，解壓後的文件夾名稱為cuda ,將對應文件複製到/usr/local中的cuda內
#下載的cuDNN library 選擇For Linux版本即可

sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

```


# Miniconda 安裝

Linux/macOS用戶:

第一步，根據操作系統下載Miniconda（網址：https://conda.io/miniconda.html）

它是一個sh文件。然後打開Terminal應用進入命令行來執行這個sh文件


```bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

```

安裝時會顯示使用條款，按“↓”繼續閱讀，按“Q”退出閱讀。之後需要回答下面幾個問題：

Do you accept the license terms? [yes|no]

[no] >>> yes

Do you wish the installer to prepend the Miniconda3 install location

to PATH in your /home/your_name/.conda ? [yes|no]

[no] >>> yes



# 後記

網路上對於ubuntu 18.04的文章很多都良莠不齊，內文很多參雜16.04 17.04的方法

經測設後缺少部分方法，以至於無法順利安裝，多數教學也捨棄使用官方提供的NVIDIA driver

一些細節上的設定也未說明清楚，在重灌10多次後，才終於排除所有的問題架設好平台

此方法為了穩定性，都使用官方所提供driver 和技術方法進行安裝。


在進行更換NVIDIA driver後，網上有些遠端安裝方式在更換後會無法進行遠端，遠端平台請參閱遠端安裝文件






















