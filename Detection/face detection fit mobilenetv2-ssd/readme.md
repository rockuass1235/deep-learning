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


![image](https://github.com/rockuass1235/deep-learning/blob/master/images/mobilenet-ssd.jpg)


