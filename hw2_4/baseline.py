###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################
import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import alexnet
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
'''
AlexNet Pre-trained
Sequential(
  (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (4): ReLU(inplace=True)
  (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU(inplace=True)
  (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (9): ReLU(inplace=True)
  (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
)
'''
def get_dataloader(folder,batch_size=32):
    # Data preprocessing
    trans = transforms.Compose([
        #transforms.Grayscale(), 
        #transforms.Resize(256),        
        #transforms.CenterCrop(224),  # crop the image to 224*224 pixels about the center
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_path, test_path = os.path.join(folder,'train'), os.path.join(folder,'valid')
    # Get dataset using pytorch functions
    train_set = ImageFolder(train_path, transform=trans)
    test_set =  ImageFolder(test_path,  transform=trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total testing batch number: {}'.format(len(test_loader)))
    return train_loader, test_loader

if __name__ == "__main__":
    # TODO
    folder, output_file = sys.argv[1], sys.argv[2]
    train_loader, val_loader = get_dataloader(folder, batch_size=32)
    #print("train_loader: ", train_loader)
    
    use_cuda = torch.cuda.is_available()
    
    extractor = alexnet(pretrained=True).features
    extractor.eval()
    features = []
    labels = []
    vali_features = []
    vali_labels = []

    with torch.no_grad(): 
        for batch, (x, label) in enumerate(train_loader,1):

            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            feat = extractor(x).view(x.size(0), 256, -1)
            feat = torch.mean(feat, 2)
            feat = feat.cpu().numpy()
            label = label.numpy()
            for f, l in zip(feat, label):
                features.append(f)
                labels.append(l)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features, labels)
    
    ################
    ## Validation ##
    ################
    with torch.no_grad():
         for batch, (x, label) in enumerate(val_loader,1):

            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            feat = extractor(x).view(x.size(0), 256, -1)
            feat = torch.mean(feat, 2)
            feat = feat.cpu().numpy()
            label = label.numpy()
            for f, l in zip(feat, label):
                vali_features.append(f)
                vali_labels.append(l)
    cnt = 0
    for i in range(len(vali_features)):
        f = vali_features[i]
        #print(knn.predict(f.reshape(1, -1)))
        if(knn.predict(f.reshape(1, -1)) == vali_labels[i]):
            cnt += 1
    acc = cnt /len(vali_features)
    print("The accuracy of validation KNN is: ", round(acc*100, 4), "%")

    tsne_alex = TSNE(n_components=2).fit_transform(features, labels)
    tsne_alex_vali = TSNE(n_components=2).fit_transform(vali_features, vali_labels)

    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#7FFFAA', '#2F4F4F', '#FF1493']#9400D3
    
    plt.title("T-SNE of ALEX-NET in Training")
    for i in range(len(labels)):
        if(labels[i] < 10):
            plt.scatter(x=tsne_alex[i, 0], y=tsne_alex[i, 1], c=colors[labels[i]])
            #plt.annotate(labels[i], (tsne_alex[i, 0], tsne_alex[i, 1]))
    plt.savefig("t-sne_alex_train.png")
    plt.show()
    
    plt.title("T-SNE of ALEX-NET in Validation")
    for i in range(len(vali_labels)):
        if(vali_labels[i] < 10):
            plt.scatter(x=tsne_alex_vali[i, 0], y=tsne_alex_vali[i, 1], c=colors[vali_labels[i]])
            #plt.annotate(vali_labels[i], (tsne_vali[i, 0], tsne_vali[i, 1]))
    plt.savefig(output_file)
    plt.show()
