import os
import sys
from pathlib import Path

import torch
from torch import nn, optim

from cnn.image_dataset import ImageDataset
from cnn.network import CNN
from utils.constants import DATABASE_FILE, SEGMENTED_LP_DIR

import argparse


def train_and_test():
    # Create dataset from segmented images and their label
    # Because there are different size of license plate numbers, batches do not work.
    batch_size = 1
    working_dir = Path().absolute().parent
    csv_path = os.path.join(working_dir, DATABASE_FILE)
    data_path = os.path.join(working_dir, SEGMENTED_LP_DIR)
    full_dataset = ImageDataset(csv_file=csv_path, data_path=data_path)

    # Drop every invalid images where the character segmentation failed
    full_dataset.sanitize_data()

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    # Train
    for epoch in range(epochs):
        for images, labels in train_loader:

            images = images.squeeze(0)
            labels = labels.squeeze(0)
            for input_iter, label_iter in zip(images, labels):
                input_tensor = input_iter.to(device)
                label_tensor = label_iter.to(device)

                optimizer.zero_grad()
                outputs = network(input_tensor)
                loss = criterion(outputs, label_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print('Finished Training')
    torch.save(network.state_dict(), "trained_network.pt")

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:

            images = images.squeeze(0)
            labels = labels.squeeze(0)
            wrong_prediction = False
            for input_iter, label_iter in zip(images, labels):
                input_tensor = input_iter.to(device)
                label_tensor = label_iter.to(device)

                _, max_label = torch.max(label_tensor.data, 1)
                # calculate outputs by running images through the network
                outputs = network(input_tensor)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)

                if predicted != max_label:
                    wrong_prediction = True
                    break

            total += 1
            correct += 1 if not wrong_prediction else 0

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


def do_evaluation():
    batch_size = 1

    dataset = ImageDataset(csv_file=csv_path, data_path=dataset_path)

    dataset.sanitize_data()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:

            images = images.squeeze(0)
            labels = labels.squeeze(0)
            wrong_prediction = False
            for input_iter, label_iter in zip(images, labels):
                input_tensor = input_iter.to(device)
                label_tensor = label_iter.to(device)

                _, max_label = torch.max(label_tensor.data, 1)
                # calculate outputs by running images through the network
                outputs = network(input_tensor)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)

                if predicted != max_label:
                    wrong_prediction = True
                    break

            total += 1
            correct += 1 if not wrong_prediction else 0

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convolutional Neural Network for training and evaluating license plate images")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--only_eval", type=bool, default=False)
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    return parser.parse_args()

if __name__ == '__main__':
    # Select cuda device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()

    epochs = args.epochs
    only_eval = args.only_eval
    csv_path = args.csv_path
    dataset_path = args.dataset_path

    working_dir = Path().absolute().parent
    if csv_path == "":
        csv_path = os.path.join(working_dir, DATABASE_FILE)
    if dataset_path == "":
        dataset_path = os.path.join(working_dir, SEGMENTED_LP_DIR)

    # Create network
    network = CNN().to(device)

    if only_eval:
        if not os.path.exists("trained_network.pt"):
            print("No trained model found. Quitting...")
            sys.exit(0)

        network.load_state_dict(torch.load("trained_network.pt"))
        network.to(device)
        network.eval()

        do_evaluation()
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

        train_and_test()
