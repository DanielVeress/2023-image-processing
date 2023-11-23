import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import ToTensor

from cnn.image_dataset import ImageDataset
from cnn.network import CNN
from utils.constants import DATABASE_FILE, ORIENTED_IMAGE_DIR

if __name__ == '__main__':
    # Select cuda device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset from segmented images and their label
    dataset_ours = ImageDataset(f"..\\{DATABASE_FILE}", f"..\\{ORIENTED_IMAGE_DIR}")
    dataset_example = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 4

    # TODO: Replace with our dataset
    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    network = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # Train
    for epoch in range(2):

        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = network(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
