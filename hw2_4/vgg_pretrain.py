import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from sklearn.manifold import TSNE

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
    
    folder, tsne_folder = sys.argv[1], sys.argv[2]
    train_loader, val_loader = get_dataloader(folder, batch_size=32)
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda: 
        model.cuda()

    ep = 10

    train_acc, valid_acc, train_loss, valid_loss = [], [], [], []
    for epoch in range(ep):
        print('Epoch:', epoch)
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        for batch, (x, label) in enumerate(train_loader,1):
            optimizer.zero_grad()
            if use_cuda:    x, label = x.cuda(), label.cuda()

            out = model(x)
            loss = criterion(out, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))

        train_acc.append(correct_cnt / total_cnt)
        train_loss.append(total_loss / batch)

        ################
        ## Validation ##
        #################    
        model.eval()
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        with torch.no_grad():
            for batch, (x, label) in enumerate(val_loader,1):
                if use_cuda:    x, label = x.cuda(), label.cuda()
            
                out = model(x)
                loss = criterion(out, label)
                total_loss += loss.item()
                _, pred_label = torch.max(out, 1)
        
                total_cnt += x.size(0)
                correct_cnt += (pred_label == label).sum().item()

                if batch % 500 == 0 or batch == len(val_loader):
                    acc = correct_cnt / total_cnt
                    ave_loss = total_loss / batch           
                    print ('Validation batch index: {}, valid loss: {:.6f}, acc: {:.3f}'.format(
                        batch, ave_loss, acc))
            valid_acc.append(correct_cnt / total_cnt)
            valid_loss.append(total_loss / batch)    
            model.train()


    torch.save(model.state_dict(), './checkpoint/resnet.pth')

    x = np.array(range(ep), dtype=int)
    
    plt.subplot(2, 2, 1)
    plt.plot(x, train_acc, c='#0072BD')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Rate')
    plt.title('Training Accuracy(final = '+ str(round(train_acc[-1]*100))+ '%)')

    plt.subplot(2, 2, 2)
    plt.plot(x, train_loss, c='#0072BD')
    plt.xlabel('epoch')
    plt.ylabel('Loss Rate')
    plt.title('Training Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(x, valid_acc, c='#D95319')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy Rate')
    plt.title('Validation Accuracy(final = '+ str(round(valid_acc[-1]*100))+ '%)')
    
    plt.subplot(224)
    plt.plot(x, valid_loss, c='#D95319')
    plt.xlabel('epoch')
    plt.ylabel('Loss Rate')
    plt.title('Validation Loss')
    plt.tight_layout()
    plt.savefig("Ours_learning curve.png")
    plt.show()

    