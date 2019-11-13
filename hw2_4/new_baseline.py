import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18

def get_dataloader(folder,batch_size=32):
    # Data preprocessing
    trans = transforms.Compose([
        transforms.ToTensor(),       # [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    folder = sys.argv[1]
    train_loader, val_loader = get_dataloader(folder, batch_size=32)
    
    use_cuda = torch.cuda.is_available()
    #### resnet16
    extractor = resnet18(pretrained=True)
    num_ftrs = extractor.fc.in_features
    extractor.fc = nn.Linear(num_ftrs,100)
    extractor.load_state_dict(torch.load('./checkpoint/resnet.pth'))
    #extractor.cuda() 
    extractor.eval()
       
    features = []
    labels = []
    features_valid = []
    labels_valid = []
    labels_val_num = np.zeros(10)
    
    with torch.no_grad():    
        for batch, (img, label) in enumerate(train_loader,1):
            #img.cuda()
            
            #feat = extractor(img.cuda())
            
            feat = extractor(img)
            feat = feat.view(img.size(0),100,-1)
            feat = torch.mean(feat,2)
            feat = feat.cpu().numpy()
            label = label.numpy()
            for f, l in zip(feat, label):
                features.append(f)
                labels.append(l)
    #### Validation
    with torch.no_grad(): 
        for batch, (img, label) in enumerate(val_loader,1):
             
            #img.cuda()
            #feat = extractor(img.cuda())
            feat = extractor(img)
            feat = feat.view(img.size(0),100,-1)
            feat = torch.mean(feat,2)
            feat = feat.cpu().numpy()
            label = label.numpy()
            for f, l in zip(feat, label):
                features_valid.append(f)
                labels_valid.append(l)

    #### t-SNE
    tsne_own = TSNE(n_components=2).fit_transform(features, labels)
    tsne_own_vali = TSNE(n_components=2).fit_transform(features_valid, labels_valid)
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#7FFFAA', '#2F4F4F', '#FF1493']#9400D3
    
    plt.title("T-SNE of RES-NET in Training")
    for i, class_num in enumerate(labels):
        if class_num<10:
            plt.scatter(tsne_own[i,0],tsne_own[i,1], c=colors[class_num])
    plt.savefig("t-sne_resnet_train.png")
    plt.show()
    plt.close()
    
    plt.title("T-SNE of RES-NET in Validation")
    for i, class_num in enumerate(labels_valid):
        if class_num<10:
            plt.scatter(tsne_own_vali[i,0],tsne_own_vali[i,1], c=colors[class_num])
    plt.savefig("t-sne_resnet_validation.png")
    plt.show()
    plt.close()
    