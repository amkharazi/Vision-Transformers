# Check Test Plan for more details 
# Test vit-tensorized model on Tiny-Imagenet-200  dataset
# Optimizer Adam
# Tiny-Imagenet-200 dataset -> (3, 224, 224) 
########################################################

# Add all .py files to path
import sys
sys.path.append('..')

# Import Libraries
from Utils.Accuracy_measures import topk_accuracy
from Utils.TinyImageNet_loader import get_tinyimagenet_dataloaders
from Utils.Num_parameter import count_parameters
from Models.vit_original import VisionTransformer

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

    TEST_ID = 'Test_ID001'
    batch_size = 64
    n_epoch = 100

    # Set up the transforms and train/test loaders
    image_size = 224

    tiny_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size)), 
            transforms.RandomCrop(image_size, padding=5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    tiny_transform_val = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    tiny_transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    
    train_loader, test_loader, _ = get_tinyimagenet_dataloaders(
                                                        data_dir = '../datasets',
                                                        transform_train=tiny_transform_train,
                                                        transform_val=tiny_transform_val,
                                                        transform_test=tiny_transform_test,
                                                        batch_size=batch_size,
                                                        image_size=image_size)
    # Set up the vit model
    model = VisionTransformer(input_size=(batch_size,3,image_size,image_size),
                patch_size=16,
                num_classes=200,
                embed_dim=16*16*3,
                num_heads=2*2*3,
                num_layers=12,
                mlp_dim=32*32*3,
                dropout=0.1,
                bias=True,
                out_embed=True,
                device=device,
                ignore_modes=None,
                Tensorized_mlp=False).to(device)
    
    
    # Load pretrained from Tests
    
    
    num_parameters = count_parameters(model)
    print(f'This Model has {num_parameters} parameters')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    
    # Define train and test functions (use examples)
    def train_epoch(loader, epoch):
        model.train()
    
        start_time = time.time()
        running_loss = 0.0
        correct = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0} # set the initial correct count for top1-to-top5 accuracy

        for i, (inputs, targets) in enumerate(loader):
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
            # print(f'batch{i} done!')

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
    result_dir = os.path.join('../results', TEST_ID)
    result_subdir = os.path.join(result_dir, 'accuracy_stats')
    model_subdir = os.path.join(result_dir, 'model_stats')

    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(model_subdir, exist_ok=True)
    
    with open(os.path.join(result_dir, 'model_stats', 'model_info.txt'), 'a') as f:
        f.write(f'total number of parameters:\n{num_parameters}')

    # Train from Scratch - Just Train
    print(f'Training for {len(range(n_epoch))} epochs\n')
    for epoch in range(0+1,n_epoch+1):
        report_train = train_epoch(train_loader, epoch)
        # report_test = test_epoch(test_loader, epoch)
    
        report = report_train + '\n' #+ report_test + '\n\n'
        if epoch % 5 == 0:
            model_path = os.path.join(result_dir, 'model_stats', f'Model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
        with open(os.path.join(result_dir, 'accuracy_stats', 'report_train.txt'), 'a') as f:
            f.write(report)
            
    # print(f'Testing\n')
    # report_test = test_epoch(test_loader, epoch)
    # report = report_test + '\n'
    # with open(os.path.join(result_dir, 'accuracy_stats', 'report_test.txt'), 'a') as f:
    #     f.write(report)       

    
            
    
    
    

