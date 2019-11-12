import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, Fully
from data import get_dataloader
import os

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'fully':
        model = Fully()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    train_acc_lst = []
    train_loss_lst = []
    vali_acc_lst = []
    vali_loss_lst = []

    # Run any number of epochs you want
    ep = 10
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
                train_acc_lst.append(acc)
                train_loss_lst.append(ave_loss)


        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        # Record the information of correct prediction and loss
        vali_correct_cnt, vali_total_loss, vali_total_cnt = 0, 0, 0
        
        with torch.no_grad():
            # Load batch data from dataloader
            for batch, (vali_x, vali_label) in enumerate(val_loader,1):

                # Put input tensor to GPU if it's available
                if use_cuda:
                    vali_x, vali_label = vali_x.cuda(), vali_label.cuda()
                # Forward input tensor through your model
                vali_out = model(vali_x)
                # Calculate loss
                vali_loss = criterion(vali_out, vali_label)

                # Calculate the testing loss and accuracy of each iteration
                vali_total_loss += vali_loss.item()
                _, vali_pred_label = torch.max(vali_out, 1)
                vali_total_cnt += vali_x.size(0)
                vali_correct_cnt += (vali_pred_label == vali_label).sum().item()

                # Show the testing information
                if batch == len(val_loader):
                    acc = vali_correct_cnt / vali_total_cnt
                    ave_loss = vali_total_loss / batch           
                    print ('Testing batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                        batch, ave_loss, acc))
                    vali_acc_lst.append(acc)
                    vali_loss_lst.append(ave_loss)
         
        model.train()
        
    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Plot Learning Curve
    # TODO

    plt.subplot(2, 2, 1)

    plt.title("Training Accuracy(final="+ str(round(vali_acc_lst[-1]*100, 2)) + "%)")
    plt.plot(train_acc_lst, c='#0072BD')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Rate")

    plt.subplot(2, 2, 2)
    plt.title("Training Loss")
    plt.plot(train_loss_lst, c='#0072BD')
    plt.xlabel("Epoch")
    plt.ylabel("Loss Rate")


    plt.subplot(2, 2, 3)
    plt.title("Validation Accuracy(final="+ str(round(vali_acc_lst[-1]*100, 2)) + "%)")
    plt.plot(vali_acc_lst, c='#D95319')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Rate")


    plt.subplot(2, 2, 4)
    plt.title("Validation Loss")
    plt.plot(vali_loss_lst, c='#D95319')
    plt.xlabel("Epoch")
    plt.ylabel("Loss Rate")
    plt.tight_layout()

    plt.show()

    sys.exit()

