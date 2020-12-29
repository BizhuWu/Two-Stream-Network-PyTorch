from LoadUCF101Data import trainset_loader, testset_loader
from Two_Stream_Net import TwoStreamNet
import torch
import torch.optim as optim
import torch.nn.functional as F



EPOCH = 100
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
SAVE_INTERVAL = 500



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


    
twoStreamNet = TwoStreamNet().to(device)



optimizer = optim.SGD(
    params=twoStreamNet.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)



def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

    

def train(epoch, save_interval):
    iteration = 0
    twoStreamNet.train()

    for i in range(epoch):
        for index, data in enumerate(trainset_loader):
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = twoStreamNet(RGB_images, OpticalFlow_images)
            loss = F.cross_entropy(output, label)

            loss.backward()
            optimizer.step()

            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('model/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)     # OpticalFlow_ResNetModel

            iteration += 1

            print("Loss: " + str(loss.item()))
            with open('log.txt', 'a') as f:
                f.write("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "'s Loss: " + str(loss.item()) + "\n")

        test(i+1)

    save_checkpoint('model/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)


def test(i_epoch):

    twoStreamNet.eval()

    correct = 0

    with torch.no_grad():
        for index, data in enumerate(testset_loader):
            RGB_images, OpticalFlow_images, label = data

            RGB_images = RGB_images.to(device)
            OpticalFlow_images = OpticalFlow_images.to(device)
            label = label.to(device)

            output = twoStreamNet(RGB_images, OpticalFlow_images)

            max_value, max_index = output.max(1, keepdim=True)
            correct += max_index.eq(label.view_as(max_index)).sum().item()

    print("Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)))
    with open('log.txt', 'a') as f:
        f.write("Epoch " + str(i_epoch) + "'s Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)) + "\n")

if __name__ == '__main__':
    train(EPOCH, SAVE_INTERVAL)
