import sys
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
import gc
from collections import Counter

from data import VideoDataset, MultiVideoDataset
from models.FrameNet import FrameNet, MultiResFrameNet


def test(dataloader, model, device):
    top1_correct = top3_correct = total = 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch
        X = torch.squeeze(X.to(device))
        y = int(y)
        pred = model(X)
        _, pred_classes = pred.max(1)
        pred_classes = pred_classes.detach().tolist()
        pred_classes = Counter(pred_classes).most_common()
        top1_pred = pred_classes[0][0]
        top3_pred = [c[0] for c in pred_classes[:3]]
        print(top3_pred, y)
        top1_correct += y == top1_pred
        top3_correct += y in set(top3_pred)
        total += 1
    top1_accuracy = top1_correct/total
    top3_accuracy = top3_correct/total
    return top1_accuracy, top3_accuracy

def test_multires(dataloader, model, device):
    top1_correct = top3_correct = total = 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        X1, X2, y = batch
        X1 = torch.squeeze(X1.to(device))
        X2 = torch.squeeze(X2.to(device))
        y = int(y)
        pred = model(X1, X2)
        _, pred_classes = pred.max(1)
        pred_classes = pred_classes.detach().tolist()
        pred_classes = Counter(pred_classes).most_common()
        top1_pred = pred_classes[0][0]
        top3_pred = [c[0] for c in pred_classes[:3]]
        print(top3_pred, y)
        top1_correct += y == top1_pred
        top3_correct += y in set(top3_pred)
        total += 1
    top1_accuracy = top1_correct/total
    top3_accuracy = top3_correct/total
    return top1_accuracy, top3_accuracy


def main(is_multires = False):
    np.random.seed(345)
    torch.manual_seed(345)
    multires_str = '_multires' if is_multires else ''
    config = {
        "n_classes": 5,
        "batch_size": 1,
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

    video_file = '/home/gaurangajitk/DL/data/sports-video-data/test_videos.csv'
    if is_multires:
        videoset = MultiVideoDataset(pd.read_csv(video_file, usecols=['video', 'label']), data_transforms_multi_res1, data_transforms_multi_res2)
    else:
        videoset = VideoDataset(pd.read_csv(video_file, usecols=['video', 'label']), data_transforms)
    video_loader = torch.utils.data.DataLoader(videoset, batch_size=1, shuffle=True, num_workers=1)

    print('Creare Model')
    if is_multires:
        model = MultiResFrameNet(config)
        test_func = test_multires
    else:
        model = FrameNet(config)
        test_func = test

    print('Check CUDA')
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(train_on_gpu):
        print('Testing on GPU.')
    else:
        print('No GPU available, Testing on CPU.')

    gc.collect()
    torch.cuda.empty_cache()

    print('******** Testing ******')
    model.to(device)
    PATH = f'/home/gaurangajitk/DL/video-classification/model_checkpoint/best_checkpoint{multires_str}.pth'
    checkpoint = torch.load(PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    top1_accuracy, top3_accuracy = test_func(video_loader, model, device)
    print(f'The top1_accuracy is {top1_accuracy}, the top3_accuracy is {top3_accuracy}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        is_multires = model_type == 'multires'
    else:
        is_multires = False
    main(is_multires)