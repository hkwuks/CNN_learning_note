import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=20,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data",train=False,download=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=2)  # shuffle:打散

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
           "ship", "truck")


# functions to show an imag
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():

    net = Net().cuda()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练
    for epoch in range(10):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs;data is a list of [inputs,labels]
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d,%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                # 预测test样本
                total = 0
                correct = 0
                for i, test_data in enumerate(testloader):
                    with torch.no_grad():
                        test_inputs, test_labers = test_data[0].cuda(
                        ), test_data[1].cuda()
                        test_outputs = net(test_inputs)
                        predicts = torch.max(test_outputs, 1)[1]
                    total += test_labers.size(0)
                    correct += (predicts == test_labers).sum()
                accuracy = correct / total * 100
                print("Accuracy of the network on the 100 test images: {}%".
                      format(accuracy))

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    main()