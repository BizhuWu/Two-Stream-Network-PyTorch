import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from PIL import Image



TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
SAMPLE_FRAME_NUM = 10



classInd = []
with open('classInd.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        idx = line[:-1].split(' ')[0]
        className = line[:-1].split(' ')[1]
        classInd.append(className)

TrainVideoNameList = []
with open('trainlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line[:-1].split('.')[0]
        video_name = video_name.split('/')[1]
        TrainVideoNameList.append(video_name)

TestVideoNameList = []
with open('testlist01.txt', 'r') as f:
    all_Class_and_Ind = f.readlines()
    for line in all_Class_and_Ind:
        video_name = line[:-1].split('.')[0]
        video_name = video_name.split('/')[1]
        TestVideoNameList.append(video_name)





class UCF101Data(Dataset):  # define a class named MNIST
    # read all pictures' filename
    def __init__(self, RBG_root, OpticalFlow_root, isTrain, transform=None):
        # root: Dataset's filepath
        # classInd: dictionary (1 -> ApplyEyeMakeup）
        self.filenames = []
        self.transform = transform


        for i in range(0, 101):
            OpticalFlow_class_path = OpticalFlow_root + '/' + classInd[i]
            RGB_class_path = RBG_root + '/' + classInd[i]

            # only load train/test data using TrainVideoNameList/TestVideoNameList
            if isTrain:
                TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_class_path)).intersection(set(TrainVideoNameList)))
            else:
                TrainOrTest_VideoNameList = list(set(os.listdir(OpticalFlow_class_path)).intersection(set(TestVideoNameList)))


            for video_dir in os.listdir(OpticalFlow_class_path):
                if video_dir in TrainOrTest_VideoNameList:
                    single_OpticalFlow_video_path = OpticalFlow_class_path + '/' + video_dir
                    signel_RGB_video_path = RGB_class_path + '/' + video_dir

                    # load Optical Flow data
                    frame_list = os.listdir(single_OpticalFlow_video_path)
                    frame_list.sort(key=lambda x:int(x.split("_")[-2]))
                    # train all clips from each video (step = 2, because the images are x_1,y_1,x_2,y_2,...)
                    for k in range(0, len(frame_list) - SAMPLE_FRAME_NUM * 2 + 1, 2):
                        stacked_OpticalFlow_image_path = []
                        for j in range(k, k + SAMPLE_FRAME_NUM * 2):
                            OpticalFlow_image_path = single_OpticalFlow_video_path + '/' + frame_list[j]
                            stacked_OpticalFlow_image_path.append(OpticalFlow_image_path)

                        # load RGB data
                        RGB_image_path = str()
                        for image_fileName in os.listdir(signel_RGB_video_path):
                            RGB_image_path = signel_RGB_video_path + '/' + image_fileName

                        # (RGB_image_path, stacked_OpticalFlow_image_path, label)
                        self.filenames.append((RGB_image_path, stacked_OpticalFlow_image_path, i))

        self.len = len(self.filenames)



    # Get a sample from the dataset & Return an image and it's label
    def __getitem__(self, index):
        RGB_image_path, stacked_OpticalFlow_image_path, label = self.filenames[index]


        # open the optical flow image
        stacked_OpticalFlow_image = torch.empty(SAMPLE_FRAME_NUM * 2, 224, 224)
        idx = 0

        for i in stacked_OpticalFlow_image_path:
            OpticalFlow_image = Image.open(i)

            # May use transform function to transform samples
            if self.transform is not None:
                OpticalFlow_image = self.transform(OpticalFlow_image)
            stacked_OpticalFlow_image[idx, :, :] = OpticalFlow_image[0, :, :]
            idx += 1


        # open the RGB image
        RGB_image = Image.open(RGB_image_path)

        # May use transform function to transform samples
        if self.transform is not None:
            RGB_image = self.transform(RGB_image)

        return RGB_image, stacked_OpticalFlow_image, label



    # get the length of dataset
    def __len__(self):
        return self.len





# define the transformation
# PIL images -> torch tensors [0, 1]
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])


# load the UCF101 training dataset
trainset = UCF101Data(
    RBG_root='data/RGB',
    OpticalFlow_root='data/OpticalFlow',
    isTrain=True,
    transform=transform
)

# divide the dataset into batches
trainset_loader = DataLoader(
    trainset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
# print(trainset)



# load the UCF101 testing dataset
testset = UCF101Data(
    RBG_root='data/RGB',
    OpticalFlow_root='data/OpticalFlow',
    isTrain=False,
    transform=transform
)

# divide the dataset into batches
testset_loader = DataLoader(
    testset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
# print(testset)
