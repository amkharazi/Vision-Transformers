# Check Test Plan for more details 
# Test ResNet50 model on MNIST dataset
# No change to classifier - Basic Model
# Optimizer Adam - Default
# No Scheduler
# MNIST dataset -> (3, 192, 192) 
# Pretrained
# Trasfer Learning
########################################################

# Add all .py files to path
import sys
sys.path.append('..')

# Import Libraries
from Utils.Accuracy_measures import topk_accuracy
from Utils.Mnist_loader import get_mnist_dataloaders
from Utils.num_parameter import count_parameters
from Models.Resnet50 import Resnet50


import torchvision.transforms as transforms
from torch import nn
from torch import optim

import time
import torch
import os


if __name__ == '__main__':
    
    # Setup the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Device is set to : {device}')

    # Set up the transforms and train/test loaders
    image_size = 192

    mnist_transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2),
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    mnist_transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    train_loader, test_loader = get_mnist_dataloaders(
                                        data_dir = '../datasets',
                                        batch_size = 64,
                                        image_size = 192,
                                        transform_train = mnist_transform_train ,
                                        transform_test = mnist_transform_test)
    # Set up the new classifier 
    
    # Set up the model, optimizer and criterion
    model = Resnet50(pretrained=True,
                          weights_path='../weights/resnet50_weights.pth',
                          input_shape=(192,192),
                          num_classes=10,
                          avg_pool=False,
                          new_classifier=None).to(device)
    
    num_parameters = count_parameters(model)
    classifier_parameters = count_parameters(model.fc)
    print(f'This Model has {num_parameters} parameters')
    print(f'This Model has {classifier_parameters} classifier parameters')

    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Define train and test functions (use examples)
    def train_epoch(loader, epoch):
        model.train()
    
        start_time = time.time()
        running_loss = 0.0
        correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0} # set the initial correct count for top1-to-top5 accuracy

        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
            for k in accuracies:
                correct[k] += accuracies[k]['correct']

        elapsed_time = time.time() - start_time
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k]/len(loader.dataset)) for k in correct]
        avg_loss = running_loss / len(loader.dataset)
    
        report_train = f'Train epoch {epoch}: top1={top1_acc}%, top2={top2_acc}%, top3={top3_acc}%, top4={top4_acc}%, top5={top5_acc}%, loss={avg_loss}, time={elapsed_time}s'
        print(report_train)

        return report_train

    def test_epoch(loader, epoch):
        model.eval()
    
        start_time = time.time()
        running_loss = 0.0
        correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0} # set the initial correct count for top1-to-top5 accuracy

        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            accuracies = topk_accuracy(outputs, targets, topk=(1, 2, 3, 4, 5))
            for k in accuracies:
                correct[k] += accuracies[k]['correct']

        elapsed_time = time.time() - start_time
        top1_acc, top2_acc, top3_acc, top4_acc, top5_acc = [(correct[k]/len(loader.dataset)) for k in correct]
        avg_loss = running_loss / len(loader.dataset)
    
        report_test = f'Test epoch {epoch}: top1={top1_acc}%, top2={top2_acc}%, top3={top3_acc}%, top4={top4_acc}%, top5={top5_acc}%, loss={avg_loss}, time={elapsed_time}s'
        print(report_test)

        return report_test
    
    # Set up the directories to save the results
    TEST_ID = 'Test_ID01'
    result_dir = os.path.join('../results', TEST_ID)
    result_subdir = os.path.join(result_dir, 'accuracy_stats')
    model_subdir = os.path.join(result_dir, 'model_stats')

    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(model_subdir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'model_stats', 'model_info.txt'), 'a') as f:
        f.write(f'total number of parameters:\n{num_parameters}\ntotal number of classifier parameters:\n{classifier_parameters}')
    
    # Freeze Convolutional Layers
    layer = 0
    for child in model.children():
        layer+=1
        if layer < 10:
            for param in child.parameters():
                param.requires_grad = False
    
    # Train and Test The Model - Frozen Layers
    n_epoch = 30
    print(f'Training for {len(range(n_epoch))} epochs\n')
    for epoch in range(1,n_epoch+1):
        report_train = train_epoch(train_loader, epoch)
        report_test = test_epoch(test_loader, epoch)
    
        report = report_train + '\n' + report_test + '\n\n'
        if epoch % 10 == 0:
            model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
        with open(os.path.join(result_dir, 'accuracy_stats', 'report.txt'), 'a') as f:
            f.write(report)
            
    # Unfreeze all layers
    layer = 0
    for child in model.children():
        layer+=1
        if layer < 10:
            for param in child.parameters():
                param.requires_grad = True
                
    # Train and Test The Model - Unfrozen Layers - comment if not required
    n_epoch_additional = 5
    print(f'Training for Additional {len(range(n_epoch_additional))} epochs\n')
    for epoch in range(n_epoch+1,n_epoch+n_epoch_additional+1):
        report_train = train_epoch(train_loader, epoch)
        report_test = test_epoch(test_loader, epoch)
    
        report = report_train + '\n' + report_test + '\n\n'
        if epoch % 5 == 0:
            model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
        with open(os.path.join(result_dir, 'accuracy_stats', 'report.txt'), 'a') as f:
            f.write(report)
    
            
    
    
    

