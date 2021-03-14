from pathlib import Path
import LeNet
import torch
import torchvision
def main():

    Path='./cifar_net.pth'

    dataiter=iter(LeNet.testloader)
    images,labels=dataiter.next()
    # print images
    LeNet.imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % LeNet.classes[labels[j]] for j in range(4)))

    net = LeNet.Net()
    net.load_state_dict(torch.load(Path))
    outputs=net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % LeNet.classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in LeNet.testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' %
        (100 * correct / total))

if __name__ == "__main__":
    main()