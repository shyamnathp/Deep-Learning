
# coding: utf-8

# ## VGG implementation with SVM

# *Python Modules*

# In[ ]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
import sklearn.svm
from sklearn.model_selection import train_test_split, KFold
import random
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

plt.ion() 

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
else:
    print("Using CPU")


# ## Dataloader functions
# ImageFolder loads the data directly from its path. transforms are used to then compose the same into the size needed for vggnet . The data is then loaded based on the input size. 

# In[ ]:


def data_loader(data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 1 ):
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, TEST]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=1,
            shuffle=True, num_workers=1
        )
        for x in [TRAIN, TEST]
    }
    print("Data loading complete")
    return dataloaders, image_datasets
    
def update_details(image_datasets):
    #print()
    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, TEST]}

    for x in [TRAIN, TEST]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x), file = log)

    print("Classes: ", file = log)
    class_names = image_datasets[TRAIN].classes
    classification_size = len(image_datasets[TRAIN].classes)
    print(image_datasets[TRAIN].classes, file =log)
    print(classification_size)
    
    return dataset_sizes, classification_size, class_names


# ## Setting up the network
# 
# Some utility function to visualize the dataset and the model's predictions

# In[ ]:


def set_up_network(net, freeze_training = True, clip_classifier = True, classification_size = 101):
    if net == 'vgg16':
    # Load the pretrained model from pytorch
        network = models.vgg16(pretrained=True)
        print(" original vgg16", network)

        # Freeze training for all layers
        # Newly created modules have require_grad=True by default
        if freeze_training:
            for param in network.features.parameters():
                param.require_grad = False

        if clip_classifier:
            features = list(network.classifier.children())[:-1] # Remove last layer
            network.classifier = nn.Sequential(*features) # Replace the model classifier
            print(network)
    
    elif net == 'resnet34':
        #networkvgg = models.vgg16(pretrained=True)
        #print(networkvgg)
        network = models.resnet34(pretrained=True)
        #print(network)
        if freeze_training:
            print("doesnt reach")
            for param in network.parameters():
                param.require_grad = False
        
        if clip_classifier:
            print("resnet clipclassifier")
            features = list(network.children())#[:-1] # Remove last layer
            network = nn.Sequential(*features) # Replace the model classifier
    if classification_size != 1000 and clip_classifier == False:
        if(net == 'vgg16'):
          print("inside vgg")
          num_features = network.classifier[6].in_features
          features = list(network.classifier.children())[:-1] # Remove last layer
          features.extend([nn.Linear(num_features, classification_size)]) # Add our layer with 4 outputs
          network.classifier = nn.Sequential(*features) # Replace the model cla
        elif(net == 'resnet34'):
          print("inside resnet")
          num_features = network.fc.in_features
          print(num_features)
          network.fc = nn.Linear(num_features, classification_size)
    #print("modified vgg16", network)
    print(network)
    return network


# ## Task 1: Update Features
# This function updates the network output for then being able to update it for SVM layer.

# In[ ]:


def get_features(ipnet, train_batches = 10, number_of_classes = 10):

    print("getting features")
    imgfeatures = []
    imglabels = []
    if classification_size < number_of_classes:
        number_of_classes = classification_size
        print("Input size smaller at:", classification_size,". Adjusting the class to this number", file = log)
    selected_classes = random.sample(range(0,classification_size), number_of_classes)
    print("The selected classes are: ",selected_classes, file = log)
    for i, data in enumerate(dataloaders[TRAIN]):
        if i % 100 == 0:
            print("\rTraining batch {}/{}".format(i, train_batches), file=log)

        # Use half training dataset
        if i > train_batches:
            break

        inputs, labels = data
        if(labels.numpy() not in selected_classes): 
            continue
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        feature = ipnet(inputs)
#         print("The shape of output is: ", feature.shape)
#         print(labels)
        if use_gpu:
            imgfeatures.append(feature.cpu().detach().numpy().flatten())
            imglabels.append(labels.cpu().detach().numpy())
        else:
            imgfeatures.append(feature.detach().numpy().flatten())
            imglabels.append(labels.detach().numpy())
        del inputs, labels, feature
        torch.cuda.empty_cache()
    print("Features Updated")
    return imgfeatures, imglabels


# # Fit features to SVM and predict output

# In[ ]:


def fit_features_to_SVM(class_names, features, labels, train_batch_size,  K=5  ):

    print("fitting to SVM")
    kf = sklearn.model_selection.KFold(n_splits=K)
    kf.get_n_splits(features)
    scores = []
    features = np.array(features)
    labels = np.array(labels)
#     print(features.shape)
#     print(labels.shape)

    i=0
    for train, test in kf.split(features):
        i+=1
        model = sklearn.svm.SVC(C=1.0, kernel='linear') #, C=1, gamma=0)
        model.fit(features[train, :], labels[train].ravel())
        out_predict = model.predict(features[test, :])
        
        y_label = labels[test].ravel()
        if(i == K):
          print("Confusion Matrix", file=log)
          print(confusion_matrix(y_label, out_predict), file=log)  
          print("-"*30, file=log)
    #         print("Classification Report")
    #         print(classification_report(y_label,out_predict))
          
          print("List of classification Accuracy", file=log)
          data = Counter(y_label[y_label==out_predict])
          stat = data.most_common()
          stat = np.array(stat)
          print(stat, file=log)   # Returns all unique items and their counts
          
          print(" The best classification accuracy is: ", stat[0,1]/np.sum(y_label[y_label==stat[0,1]]), file=log)
          print(" The worst classification accuracy is: ", stat[-1,1]/np.sum(y_label[y_label==stat[-1,1]]), file=log)
        
        s=model.score(features[test, :], labels[test])
        print(i,"/",K,"The score for this classification is: ", s, file = log)
        scores.append(s)
    return np.mean(scores), np.std(scores)


# This is an alternative implementation using the same thing.


# ## VGG16 implementation with SVM as a classification layer. (All Updates here)
# This updates the data, sets up the network and classifies using SVM.

data_dir_10 = "/content/Deep_Learning_Class10/class10"
data_dir_30 = "/content/Deep_Learning_Class10/class30"
TRAIN = 'train'
TEST = 'test'
log = open("VGG16_Task1_RESNET.txt", "w")
vgg16_nc = set_up_network('vgg16', clip_classifier = False, freeze_training = True)
if use_gpu:
    vgg16_nc.cuda() #.cuda() will move everything to the GPU side

ImageDirectory = [data_dir_10, data_dir_30]
mean_accuracy_of_5_splits=0.0
k=3
for data_dir in ImageDirectory:
    
    print(data_dir)
    # Get Data
    dataloaders, image_datasets = data_loader(data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 1 )
    dataset_sizes, classification_size, class_names = update_details(image_datasets)
    
    # Update train_batch_size
    train_batch_size = dataset_sizes[TRAIN]
#     train_batch_size = 10
    class_size = classification_size
    
    # Get the image features for the imagenet trained network.
    imgfeatures_vgg, imglabels_vgg = get_features(vgg16_nc, train_batch_size, number_of_classes = class_size)
    mean_accuracy, sd = fit_features_to_SVM(class_names,imgfeatures_vgg,
                                        imglabels_vgg, train_batch_size, K=k )
    mean_accuracy_of_5_splits+=mean_accuracy
    print("The mean and standard deviation of classification for vgg 16 is: ",
      mean_accuracy, sd, "for class size: ", class_size, file = log)
    k=k-1
    del dataloaders, image_datasets, imgfeatures_vgg, imglabels_vgg
del vgg16_nc
print("Average Classification accuracy over 5 splits for vgg16 : " + str(mean_accuracy_of_5_splits/5.0))

mean_accuracy_of_5_splits=0.0
k=3
for data_dir in ImageDirectory:
    
    # Get Data
    dataloaders, image_datasets = data_loader(data_dir, TRAIN, TEST, image_crop_size = 224, mini_batch_size = 1 )
    dataset_sizes, classification_size, class_names = update_details(image_datasets)

    resnet34_nc = set_up_network('resnet34', clip_classifier=False, freeze_training = True, classification_size=classification_size)
    if use_gpu:
      resnet34_nc.to(torch.device("cuda")) #.cuda() will move everything to the GPU side

    # Update train_batch_size
    train_batch_size = dataset_sizes[TRAIN]
#     train_batch_size = 10
    class_size = classification_size
    
    # Get the image features for the imagenet trained network.
    imgfeatures_res, imglabels_res = get_features(resnet34_nc, train_batch_size, number_of_classes = class_size)
    mean_accuracy, sd = fit_features_to_SVM(class_names,imgfeatures_res,
                                        imglabels_res, train_batch_size, K=k )
    mean_accuracy_of_5_splits+=mean_accuracy
    print("The mean and standard deviation of classification for resnet 34 is: ",
      mean_accuracy, sd, "for class size: ", class_size, file = log)
    k=k-1
    del dataloaders, image_datasets, imgfeatures_res, imglabels_res
del resnet34_nc
print("Average Classification accuracy over splits for resnet 34 : " + str(mean_accuracy_of_5_splits/5.0))
log.close()
