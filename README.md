# Two-Stream Network implemented in PyTorch
Paper's Link：[Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/pdf/1604.06573.pdf)

The backbone of each stream is **`ResNet 50`**

&nbsp;


## Performance
Stream     | Accuracy
:-----------:|:-----------:
RGB  | -
Optical Flow  | -
Fusion (Two Stream)  | **`73.53%`** (only stack 4 optical flow images：2 x_direction 2 y_direction)

&nbsp;


## Training Environment
+ Ubuntu 16.04.7 LTS
+ CUDA Version: 10.1
+ PyTorch 1.3.1
+ torchvision 0.4.2
+ numpy 1.19.2
+ pillow 8.0.1
+ python 3.6.12

&nbsp;


## Data Preparation
Original Dataset：[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

By the way, I write a matlab code to generate the **optical flow** images and the **RGB** images.

+ For the **optical flow** images, I call the **`Horn–Schunck Algorithm`** function in matlab to calculate it. The video frame interval for calculating the optical flow images is set to **`2`** to generate sufficient data.

+ For the **RGB** images, I just randomly sampled **`one`** single frame from each video.

Generating Data Code (Matlab)：[calOpticalFlow.m](https://github.com/BizhuWu/Two-Stream-Network-PyTorch/blob/main/calOpticalFlow.m)

downloading processed data：[Link](https://pan.baidu.com/s/1RH62JS04nqo6URTojvDnCA) password：rhf1 

&nbsp;


## Train
Before training, you should new a directory named **`model`** to save checkpoint file.
```python
python3 trainTwoStreamNet.py
```
&nbsp;


## Problems
I recorded some problems and solutions when writing the code. Really so sorry that I only write in Chinese! 
Here is the [Link](https://blog.csdn.net/qq_36627158/article/details/110765411)
