# Two-Stream Network implemented in PyTorch
Paper's Link：[Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/pdf/1604.06573.pdf)

The backbone of each stream is **`ResNet-50`**

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
### The First Way
Original Dataset：[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

or

### The Second Way
By the way, I write a matlab code to generate the **optical flow** images and the **RGB** images.

+ For the **optical flow** images, I call the **`Horn–Schunck Algorithm`** function in matlab to calculate it. The video frame interval for calculating the optical flow images is set to **`2`** to generate sufficient data.

+ For the **RGB** images, I just randomly sampled **`one`** single frame from each video.

Generating Data Code (Matlab)：[calOpticalFlow.m](https://github.com/BizhuWu/Two-Stream-Network-PyTorch/blob/main/calOpticalFlow.m)

downloading processed data：[Link](https://pan.baidu.com/s/1AV0Bp7nt17QKTHM8KUiROA) password：peyu

After downloading processed data, you should unrar the **`processedData.rar`** and build a directory named **`data`**
```
Project
│--- data
│------ RGB
│------ OpticalFlow
│--- other files
```

&nbsp;


## Train
Before training, you should new a directory named **`model`** to save checkpoint file.
```python
python3 trainTwoStreamNet.py
```
&nbsp;


## demo
This is a demo video for test. I randomly set the **`test_video_id = 1000`** from **`testset`** to run this demo python file. What's more, I use the checkpoint file saved in **`9000-th`** iteration as the demo model.

You can change the **`test_video_id`** at here:
```python
# set the test video id in testset
test_video_id = 1000
print('Video Name:', LoadUCF101Data.TestVideoNameList[test_video_id])
```

You can change the **`checkpoint_file_path`** at here:
```python
# load the chekpoint file
state = torch.load('model/checkpoint-9000.pth')
twoStreamNet.load_state_dict(state['model'])
```

run **`demo.py`** file
```python
CUDA_VISIBLE_DEVICES=0 python3 demo.py
```
output:

![demo_RGB](https://github.com/BizhuWu/Two-Stream-Network-PyTorch/blob/main/demo_RGB.png)

![demo_stackedOpticalFlowImg](https://github.com/BizhuWu/Two-Stream-Network-PyTorch/blob/main/demo_opticalFlowStackedImgs.png)
```
Video Name: v_Drumming_g01_c05
actual class is Drumming
predicted class is Drumming , probability is 99.9534
```
&nbsp;


## Problems
I recorded some problems and solutions when writing the code. Really so sorry that I only write in Chinese! 
Here is the [Link](https://blog.csdn.net/qq_36627158/article/details/110765411)
