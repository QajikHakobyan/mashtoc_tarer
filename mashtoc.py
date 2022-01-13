from torch.optim import optimizer
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torchvision import models
from ConvLayers import ConvLayers
from Spinal import Spinal

num_epochs = 200
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.5
log_interval = 500
device = 'cuda'

def train(model, folder):
    global num_epochs, batch_size_test, batch_size_train, learning_rate, momentum, log_interval, device
    train_transforms = torchvision.transforms.Compose([
                                torchvision.transforms.RandomPerspective(), 
                                torchvision.transforms.RandomRotation(10, fill=(0, 0, 0)), 
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
    train_dataset = torchvision.datasets.ImageFolder(root="/ssd/tarer/Train" ,transform=train_transforms)

    test_transforms = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])
    test_dataset = torchvision.datasets.ImageFolder(root="/ssd/tarer/Test",transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)
    model.to(device=device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    def update_lr(optimizer, ratio):
        lr = 0  
        for param_group in optimizer.param_groups:
            if param_group['lr'] <= 1e-6:
                return
            param_group['lr'] *= ratio
            lr = param_group['lr']
        print("updateing learning rate =", lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    total_step = len(train_loader)
    # net_opt = None
    best_accuracy = 0
    idx = 0
    losses = []
    test_score = []

    for epoch in range(num_epochs):
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            # print(torch.max(images))
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0:
                losses += [loss]

            if i == 616:
                print ("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        

            
        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
            test_score += [correct / total]
            
            if best_accuracy >= correct / total:
                # ratio = np.asscalar(pow(np.random.rand(1), 3))
                idx += 1
                if idx >= 10:
                    idx = 0
                    # print("updateing learning rate")
                    update_lr(optimizer, 0.5)
                print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct / total, 100*best_accuracy))
            else:
                idx = 0
                best_accuracy = correct / total
                torch.save(model.state_dict(), f"{folder}/model_{best_accuracy}.pth")            
                print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct / total))

            
                
            model.train()

    torch.save(model.state_dict(), f"{folder}/model.pth")

    import pickle
    file_to_store = open(f"{folder}/losses.pickle", "wb")
    pickle.dump(losses, file_to_store)

    file_to_store = open(f"{folder}/test_score.pickle", "wb")
    pickle.dump(test_score, file_to_store)


model = nn.Sequential(
    ConvLayers(),
    nn.Sequential(
        nn.Dropout(p = 0.5),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p = 0.5),
        nn.Linear(512, 78),
    )
)
train(model, "/ssd/tarer/models/vgg_small")

model = nn.Sequential(
    ConvLayers(),
    Spinal(num_classes=78, layer_width=128, half_width=128)
)
train(model, "/ssd/tarer/models/vgg_small_spinal")


model = models.vgg16().to(device=device)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
model.classifier = nn.Sequential(
                    nn.Dropout(p = 0.5),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p = 0.5),
                    nn.Linear(512, 78),
                )
train(model, "/ssd/tarer/models/vgg")

model = models.vgg16().to(device=device)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
model.classifier = Spinal(78, 256, 256)
train(model, "/ssd/tarer/models/vgg_spinal")

model = models.resnet18().to(device=device)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Sequential(
                    nn.Dropout(p = 0.5),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p = 0.5),
                    nn.Linear(512, 78),
                )
train(model, "/ssd/tarer/models/resnet")

model = models.resnet18().to(device=device)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = Spinal(78, 256, 256)
train(model, "/ssd/tarer/models/resnet_spinal")
