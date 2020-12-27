from PIL import Image
import matplotlib.pyplot as plt
from Two_Stream_Net import TwoStreamNet
from LoadUCF101Data import testset
import LoadUCF101Data
import torch
import torch.nn.functional as F
import numpy as np



# setting gpu or cpu
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



# new a Two Stream model
twoStreamNet = TwoStreamNet().to(device)



if __name__ == '__main__':
    # set the test video id in testset
    test_video_id = 1000
    print('Video Name:', LoadUCF101Data.TestVideoNameList[test_video_id])



    # load the chekpoint file
    state = torch.load('model/checkpoint-9000.pth')
    twoStreamNet.load_state_dict(state['model'])



    # send the model into the device, set its mode to eval
    twoStreamNet = twoStreamNet.to(device)
    twoStreamNet.eval()



    # get demo video's img and stacked optical flow images, show them in plt
    demo_RGB_img_path, demo_StackedOpticalFlow_imgs_path, label = testset.filenames[test_video_id]

    RGB_img = Image.open(demo_RGB_img_path)
    plt.figure()
    plt.axis('off')
    plt.imshow(RGB_img)
    plt.show()

    plt.figure()
    for i in range(0, LoadUCF101Data.SAMPLE_FRAME_NUM * 2):
        plt.subplot(4, np.ceil(LoadUCF101Data.SAMPLE_FRAME_NUM * 2 / 4), i+1)
        plt.axis('off')
        opticalFlow_i_img = Image.open(demo_StackedOpticalFlow_imgs_path[i])
        plt.imshow(opticalFlow_i_img)
    plt.show()



    # send demo video's img and stacked optical flow images into the model
    RGB_img, opticalFlowStackedImg, actual_label = testset[test_video_id]

    RGB_img = RGB_img.to(device)
    opticalFlowStackedImg = opticalFlowStackedImg.to(device)

    RGB_img = RGB_img.unsqueeze(0)
    opticalFlowStackedImg = opticalFlowStackedImg.unsqueeze(0)

    output = twoStreamNet(RGB_img, opticalFlowStackedImg)



    # get the most possible result
    prob = F.softmax(output, dim=1)
    max_value, max_index = torch.max(prob, 1)
    pred_class = max_index.item()
    print('actual class is', LoadUCF101Data.classInd[actual_label])
    print('predicted class is',  LoadUCF101Data.classInd[pred_class], ', probability is', round(max_value.item(), 6) * 100)
