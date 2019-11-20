import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from load_mnist import * 
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=True)

input_size = 784
hidden_sizes = [512, 512]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[1], output_size))
                    # nn.Softmax(dim=1))

criterion = nn.CrossEntropyLoss()

#time0 = time()
def full_operation(optimizer, model, l=3):
    epochs = 25
    for e in range(epochs):
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            l1_regularization = torch.tensor(0)
            l2_regularization = torch.tensor(0)

            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            cross_entropy_loss = criterion(output, labels)

            for param in model.parameters():
                l1_regularization += torch.norm(param, 1)
                l2_regularization += torch.norm(param, 2)
         
            if(l == 1):
                loss = cross_entropy_loss + l1_regularization
            elif (l == 2):
                loss = cross_entropy_loss + l2_regularization
            else:
                loss = cross_entropy_loss
            
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
        #     running_loss += loss.item()
        # else:
        #     print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

    correct_count, all_count = 0, 0
    for images,labels in valloader:
      for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
optimizerOne = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, nesterov = True)

full_operation(optimizer, model)
full_operation(optimizerOne, model)
#L1
full_operation(optimizerOne, model, 1)
#L2
full_operation(optimizerOne, model, 2)

