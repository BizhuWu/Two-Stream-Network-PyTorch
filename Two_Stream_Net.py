import torch.nn as nn
import torchvision.models as models
import LoadUCF101Data



class OpticalFlowStreamNet(nn.Module):
    def __init__(self):
        super(OpticalFlowStreamNet, self).__init__()

        self.OpticalFlow_stream = models.resnet50(pretrained=True)
        self.OpticalFlow_stream.conv1 = nn.Conv2d(LoadUCF101Data.SAMPLE_FRAME_NUM * 2, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.OpticalFlow_stream.fc = nn.Linear(in_features=2048, out_features=101)

    def forward(self, x):
        streamOpticalFlow_out = self.OpticalFlow_stream(x)
        return streamOpticalFlow_out



class RGBStreamNet(nn.Module):
    def __init__(self):
        super(RGBStreamNet, self).__init__()

        self.RGB_stream = models.resnet50(pretrained=True)
        self.RGB_stream.fc = nn.Linear(in_features=2048, out_features=101)

    def forward(self, x):
        streamRGB_out = self.RGB_stream(x)
        return streamRGB_out



class TwoStreamNet(nn.Module):
    def __init__(self):
        super(TwoStreamNet, self).__init__()

        self.rgb_branch = RGBStreamNet()
        self.opticalFlow_branch = OpticalFlowStreamNet()

    def forward(self, x_rgb, x_opticalFlow):
        rgb_out = self.rgb_branch(x_rgb)
        opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)
        return rgb_out + opticalFlow_out