import pandas as pd
import torch
from torchvision import transforms
import time
import  numpy as np
import logging
import os
import sys
import gc
from matplotlib import pyplot as plt

from data import FrameDataset, MultiFrameDataset
from utils import EarlyStopping, LRScheduler
from models.FrameNet import FrameNet, MultiResFrameNet


def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)

def train_multires(dataloader, model, criterion, optimizer, device):
    loss_epoch = correct = total = 0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X1, X2, y = batch
        X1 = X1.to(device)
        X2 = X2.to(device)
        y = y.to(device)
        pred = model(X1, X2)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
        optimizer.zero_grad()
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy

def test_multires(dataloader, model, criterion, device):
    loss_epoch = correct = total = 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        X1, X2, y = batch
        X1 = X1.to(device)
        X2 = X2.to(device)
        y = y.to(device)
        pred = model(X1, X2)
        loss = criterion(pred, y)
        loss_epoch += loss.item()
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy

def train(dataloader, model, criterion, optimizer, device):
    loss_epoch = correct = total = 0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
        optimizer.zero_grad()
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy

def test(dataloader, model, criterion, device):
    loss_epoch = correct = total = 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        loss_epoch += loss.item()
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
    accuracy = correct/total
    avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy


def main(is_multires = False):
    start = time.time()
    np.random.seed(345)
    torch.manual_seed(345)
    multires_str = '_multires' if is_multires else ''
    logging.basicConfig(filename=f'./logs/training{multires_str}.log', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info('\n************************************\n')
    print('\n************************************\n')
    num_classes = 5

    config = {
        "n_classes": 5,
        "batch_size": 1024,
        "lr": 1e-3,
        "gradient_clip_val": 0.5,
        "num_epochs": 50,
        "cnn1_in": 3,
        "cnn2_in": 96,
        "cnn3_in": 256,
        "cnn4_in": 384,
        "cnn5_in": 384,
        "cnn5_out": 256,
        "linear_in": 4096,
        "dropout": 0.5,
        "weight_decay": 5e-4
    }

    input_size = 170
    multi_res_input = 89

    print(config)
    logging.info(config)
    print('Preparing Datasets')

    data_transforms = transforms.Compose([
        transforms.ToTensor(),\
        transforms.Resize(size=(input_size, input_size), interpolation=transforms.functional.InterpolationMode.NEAREST),\
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), \
        transforms.RandomHorizontalFlip()
         ])
    
    data_transforms_multi_res1 = transforms.Compose([
        transforms.ToTensor(),\
        transforms.Resize(size=(multi_res_input, multi_res_input), interpolation=transforms.functional.InterpolationMode.NEAREST),\
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), \
        transforms.RandomHorizontalFlip()
         ])
    
    data_transforms_multi_res2 = transforms.Compose([
        transforms.ToTensor(),\
        transforms.CenterCrop(multi_res_input),\
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), \
        transforms.RandomHorizontalFlip()
         ])

    train_file = '/home/gaurangajitk/DL/data/sports-video-data/train_images.csv'
    test_file = '/home/gaurangajitk/DL/data/sports-video-data/test_images.csv'

    if is_multires:
        trainset = MultiFrameDataset(pd.read_csv(train_file, usecols=['frame', 'label']), data_transforms_multi_res1, data_transforms_multi_res2)
        testset = MultiFrameDataset(pd.read_csv(test_file, usecols=['frame', 'label']), data_transforms_multi_res1, data_transforms_multi_res2)
    else:
        trainset = FrameDataset(pd.read_csv(train_file, usecols=['frame', 'label']), data_transforms)
        testset = FrameDataset(pd.read_csv(test_file, usecols=['frame', 'label']), data_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    print('Creare Model')
    if is_multires:
        model = MultiResFrameNet(config)
        train_func = train_multires
        test_func = test_multires
    else:
        model = FrameNet(config)
        train_func = train
        test_func = test

    print('Setup criterion and optimizer')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    print('Check CUDA')
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    
    print('***** Training *****')
    logging.info('Started Training')

    model.to(device)

    best_valid_acc = 0
    train_history_loss = []
    train_history_acc = []
    val_history_loss = []
    val_history_acc = []

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        train_loss, train_acc = train_func(train_loader, model, criterion, optimizer, device)
        valid_loss, valid_acc = test_func(test_loader, model, criterion, device)
        time_for_epoch = time.time() - epoch_start

        print(f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, Valid Acc= {valid_acc:.3f} \t Time Taken={time_for_epoch:.2f} s')
        logging.info(
            f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, Valid Acc= {valid_acc:.3f} \t Time Taken={time_for_epoch:.2f} s')
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            checkpoint = {
                        'epoch': epoch, 
                        'model': model.state_dict(), 
                        'criterion': criterion.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_valid_acc
                        }
            save_checkpoint(checkpoint, directory='./model_checkpoint', file_name=f'best_checkpoint{multires_str}')
            logging.info(f'Checkpoint saved at Epoch {epoch}')

        lr_scheduler(valid_loss)
        early_stopping(valid_loss)
        #save losses for learning curves
        train_history_loss.append(train_loss)
        val_history_loss.append(valid_loss)
        train_history_acc.append(train_acc)
        val_history_acc.append(valid_acc)
        if early_stopping.early_stop:
            break
    del model; del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    logging.info(f'Final scheduler state {lr_scheduler.get_final_lr()}\n')

    # save curves
    plt.plot(range(len(train_history_loss)),train_history_loss, label="Training")
    plt.plot(range(len(val_history_loss)),val_history_loss, label="Validation")
    plt.legend()
    plt.title(f"Loss Curves{multires_str}")
    plt.savefig(f'curves/loss_curves{multires_str}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    plt.plot(range(len(train_history_acc)),train_history_acc, label="Training")
    plt.plot(range(len(val_history_acc)),val_history_acc, label="Validation")
    plt.legend()
    plt.title(f"Accuracy Curves{multires_str}")
    plt.savefig(f'curves/acc_curves{multires_str}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    # end
    diff = time.time() - start
    logging.info(f'Total time taken= {str(diff)} s')
    print(f'Total time taken= {str(diff)} s')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        is_multires = model_type == 'multires'
    else:
        is_multires = False
    main(is_multires)