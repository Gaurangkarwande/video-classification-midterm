import cv2
from torch.utils.data import Dataset
import torch

class FrameDataset(Dataset):
    def __init__(self, csv_file, data_transform) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.data_transform = data_transform
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        item = self.csv_file.iloc[idx]
        X = cv2.imread(item.frame)
        y = item.label

        X = self.data_transform(X)

        return X, y

class MultiFrameDataset(Dataset):
    def __init__(self, csv_file, data_transform1, data_transform2) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.data_transform1 = data_transform1
        self.data_transform2 = data_transform2
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        item = self.csv_file.iloc[idx]
        X = cv2.imread(item.frame)
        y = item.label

        X1 = self.data_transform1(X)
        X2 = self.data_transform2(X)

        return X1, X2, y

class VideoDataset(Dataset):
    def __init__(self, csv_file, data_transform) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.data_transform = data_transform
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        item = self.csv_file.iloc[idx]
        cap = cv2.VideoCapture(item.video)
        y = item.label
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.data_transform(frame)
            frames.append(frame)
        cap.release()
        X = torch.stack(frames)
        return X, y

class MultiVideoDataset(Dataset):
    def __init__(self, csv_file, data_transform1, data_transform2) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.data_transform1 = data_transform1
        self.data_transform2 = data_transform2
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        item = self.csv_file.iloc[idx]
        cap = cv2.VideoCapture(item.video)
        y = item.label
        frames1 = []
        frames2 = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame1 = self.data_transform1(frame)
            frame2 = self.data_transform2(frame)
            frames1.append(frame1)
            frames2.append(frame2)
        cap.release()
        X1 = torch.stack(frames1)
        X2 = torch.stack(frames2)
        return X1, X2, y




