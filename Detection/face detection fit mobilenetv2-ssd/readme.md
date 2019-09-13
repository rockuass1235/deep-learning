# 為何使用 MobileNet?


mobile net 使用了深度可分離卷積（Depth-wise Separable Convolution）的技術，在相同輸出下有效的降低類神經網路的參數量

讓模型的計算量縮小，可以將計算放在更小規模的誌算機上


# MobileNet V2 架構

```
MobileNetV2(
  (features): HybridSequential(
    (0): Conv2D(3 -> 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
    (2): RELU6(
    
    )
    (3): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(32 -> 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
        (5): RELU6(
        
        )
        (6): Conv2D(32 -> 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=16)
      )
    )
    (4): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(16 -> 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=96)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=96)
        (5): RELU6(
        
        )
        (6): Conv2D(96 -> 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=24)
      )
    )
    (5): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(24 -> 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=144)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=144)
        (5): RELU6(
        
        )
        (6): Conv2D(144 -> 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=24)
      )
    )
    (6): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(24 -> 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=144)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=144)
        (5): RELU6(
        
        )
        (6): Conv2D(144 -> 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
      )
    )
    (7): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(32 -> 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=192)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=192)
        (5): RELU6(
        
        )
        (6): Conv2D(192 -> 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
      )
    )
    (8): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(32 -> 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=192)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=192)
        (5): RELU6(
        
        )
        (6): Conv2D(192 -> 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=32)
      )
    )
    (9): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(32 -> 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=192)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=192)
        (5): RELU6(
        
        )
        (6): Conv2D(192 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
      )
    )
    (10): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(64 -> 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (5): RELU6(
        
        )
        (6): Conv2D(384 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
      )
    )
    (11): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(64 -> 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (5): RELU6(
        
        )
        (6): Conv2D(384 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
      )
    )
    (12): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(64 -> 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (5): RELU6(
        
        )
        (6): Conv2D(384 -> 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)
      )
    )
    (13): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(64 -> 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=384)
        (5): RELU6(
        
        )
        (6): Conv2D(384 -> 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=96)
      )
    )
    (14): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(96 -> 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=576)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=576)
        (5): RELU6(
        
        )
        (6): Conv2D(576 -> 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=96)
      )
    )
    (15): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(96 -> 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=576)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=576)
        (5): RELU6(
        
        )
        (6): Conv2D(576 -> 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=96)
      )
    )
    (16): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(96 -> 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=576)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=576)
        (5): RELU6(
        
        )
        (6): Conv2D(576 -> 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=160)
      )
    )
    (17): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(160 -> 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=960)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=960)
        (5): RELU6(
        
        )
        (6): Conv2D(960 -> 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=160)
      )
    )
    (18): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(160 -> 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=960)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=960)
        (5): RELU6(
        
        )
        (6): Conv2D(960 -> 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=160)
      )
    )
    (19): LinearBottleneck(
      (out): HybridSequential(
        (0): Conv2D(160 -> 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=960)
        (2): RELU6(
        
        )
        (3): Conv2D(1 -> 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=960)
        (5): RELU6(
        
        )
        (6): Conv2D(960 -> 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=320)
      )
    )
    (20): Conv2D(320 -> 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (21): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=1280)
    (22): RELU6(
    
    )
    (23): GlobalAvgPool2D(size=(1, 1), stride=(1, 1), padding=(0, 0), ceil_mode=True)
  )
  (output): HybridSequential(
    (0): Conv2D(None -> 1000, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): Flatten
  )
)
```

以下是每一層輸出的特徵圖大小:

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/mobilenet_v2_shape.png)






# 設計 MobileNetV2-SSD

首先參考了以下對於mobilenet v1-SSD的設計架構:

他對於mobilenet v1 後面多接了幾層 down sample blk 做特徵分析， **為什麼要修改原有架構可是VGG16-SSD就不用呢?**


VGG16是從Conv4_3也就是第10層卷積層取出38x38分辨率的特徵圖， 再觀察一下MobileNet v1-300的模型，想要取出38x38分辨率的特徵圖，最深也只能從Conv5也就是第6層卷積層取出

這個位置比較淺，實在很難保證網絡提取出了足夠有用的特徵可以使用。因此作者選擇從 19x19 以後開始取出，並在後面做延伸增加特徵圖，這也導致可檢測的物件大小與精度受影響，但好處是比VGG快約3倍

mobilenet v2對模型做了一些修正，在38x38的特徵圖上已經到Conv9，因此在設計上我選擇則從 38x38開始，為了減少計算量 將後面通道數1280全部刪除，接上簡單的特徵擷取blk


![image](https://github.com/rockuass1235/deep-learning/blob/master/images/mobilenet-ssd.jpg)


```

class MyMod(nn.Block):

    def __init__(self, classes, ctx = mx.cpu(), **kwargs):
        super(MyMod, self).__init__(**kwargs)
        self.classes = classes
        self.ctx = ctx

        net = gluon.model_zoo.vision.mobilenet_v2_1_0(pretrained=True).features
        n = (1.05-0.1)/6
        
        self.net_0 = MyBlk(net[:9], classes, [0.1, ((0.1)*(0.1+n))**0.5], [0.5, 1, 2])
        self.net_1 = MyBlk(net[9:16], classes, [0.1+n, ((0.1+n)*(0.1+2*n))**0.5], [0.5, 1, 2])
        self.net_2 = MyBlk(net[16:20], classes, [0.1+2*n, ((0.1+2*n)*(0.1+3*n))**0.5], [0.5, 1, 2])
        self.net_3 = MyBlk(down_sample_blk(512), classes, [0.1+3*n, ((0.1+3*n)*(0.1+4*n))**0.5], [0.5, 1, 2])
        self.net_4 = MyBlk(down_sample_blk(256), classes, [0.1+4*n, ((0.1+4*n)*(0.1+5*n))**0.5], [0.5, 1, 2])
        self.net_5 = MyBlk(nn.GlobalMaxPool2D(), classes, [0.1+5*n, ((0.1+5*n)*(0.1+6*n))**0.5], [0.5, 1, 2])
       
        

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

```


#結果

測試訓練結果:

```
epoch: 68, acc: 0.999372, mae loss: 0.000662, time: 179.096 sec
epoch: 69, acc: 0.999370, mae loss: 0.000661, time: 179.349 sec
epoch: 70, acc: 0.999375, mae loss: 0.000660, time: 179.758 sec
epoch: 71, acc: 0.999375, mae loss: 0.000658, time: 180.159 sec
epoch: 72, acc: 0.999381, mae loss: 0.000657, time: 179.830 sec
```

類別預測的準確率很高，但是臉部常常用矩形框取代正方形框。

將會在下一次修正以增加大小數量， 並將anchor 設定只有正方形來嘗試

![image](https://github.com/rockuass1235/deep-learning/blob/master/images/mobilenet_sample1.png)



# 資料來源

http://zh.gluon.ai/chapter_computer-vision/ssd.html

https://hey-yahei.cn/2018/08/08/MobileNets-SSD/index.html






