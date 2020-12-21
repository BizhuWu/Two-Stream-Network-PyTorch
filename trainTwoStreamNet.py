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


# vis = visdom.Visdom(server='http://127.0.0.1', port=8097)



def save_checkpoint(path, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def train(epoch, save_interval):
    x_loss, y_loss = 0, 0
    # win_loss = vis.line(X=np.array([x_loss]), Y=np.array([y_loss]), opts=(dict(title='loss')))
    x_acc, y_acc = 0, 0
    # win_Acc = vis.line(X=np.array([x_acc]), Y=np.array([y_acc]), opts=(dict(title='Accuracy')))

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

            x_loss += 1
            # vis.line(X=np.array([x_loss]), Y=np.array([loss.item()]), win=win_loss, update='append')

            print("Loss: " + str(loss.item()))
            with open('log.txt', 'a') as f:
                f.write("Epoch " + str(i+1) + ", Iteration " + str(index+1) + "'s Loss: " + str(loss.item()) + "\n")

        x_acc += 1
        test(x_acc, None)

    save_checkpoint('model/checkpoint-%i.pth' % iteration, twoStreamNet, optimizer)


def test(x_acc, win_Acc):

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

    # vis.line(X=np.array([x_acc]), Y=np.array([correct*1.0*100/len(testset_loader.dataset)]), win=win_Acc, update='append')

    print("Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)))
    with open('log.txt', 'a') as f:
        f.write("Epoch " + str(x_acc) + "'s Accuracy: " + str(correct*1.0*100/len(testset_loader.dataset)) + "\n")

if __name__ == '__main__':
    train(EPOCH, SAVE_INTERVAL)