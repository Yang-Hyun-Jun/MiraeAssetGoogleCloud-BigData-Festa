from email.mime import image
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class GenerateDataset:
    def __init__(self, size, data: pd.DataFrame):
        """
        price_data: csv with two columes Close, Volume
        """

        self.data = data
        self.size = size
        self.scaler = MinMaxScaler()

    def __fit_scaler(self):
        self.scaler.fit(self.data)
        return self.scaler.transform(self.data)       

    def __generate_wbt(self, w=5, v=4, info=True):
        """
        wb: windowed binning data
        t: trend target
        w: window size for ma
        v: continous day for trend
        """
        
        size = self.size
        # Min - Max scaling
        scaled_data = self.__fit_scaler()

        # Upper, Lower는 [0, 1]을 7등분하여 binning
        bin_num= (size-2) / 2 
        binning_data = np.zeros(shape=scaled_data.shape)
        binning_data[:,0:1] = np.array(list(map(lambda x:x//(1/(bin_num-1)), scaled_data[:,0:1])))
        binning_data[:,1:2] = np.array(list(map(lambda x:x//(1/(bin_num-1)), scaled_data[:,1:2])))

        # window 작업을 위한 인덱스 계산
        start_left = 0
        end_right = (binning_data.shape[0] - 1) - v
        end_left = end_right - size + 1
        wb_num = end_left - start_left + 1

        wb_data = np.ones(shape=(wb_num, size, 2)) * np.nan
        t_data = np.ones(shape=(wb_num, 1)) * np.nan

        # window 자르기 및 MA 기반 Trend 계산
        for i in range(wb_num):
            wb_data[i] = copy.deepcopy(binning_data[i:i+size])

            ma_data = pd.DataFrame(scaled_data[i+size-(w-1):i+size+v, 0]).rolling(w).mean()
            ma_data = list(ma_data.to_numpy().reshape(-1))[w-1:]

            if ma_data == sorted(ma_data):
                trend = 0
            elif ma_data == sorted(ma_data, reverse=True):
                trend = 1
            else:
                trend = 2

            t_data[i] = trend      

        if np.nan in wb_data.reshape(-1): 
            raise UserWarning("Invalid wb")
        if np.nan in t_data.reshape(-1) and v != 0:
            raise UserWarning("Invalid target")
        
        if info:
            print("===== Value Info =====")
            print(f"Max Close : {self.scaler.data_max_[0]}, Min Close: {self.scaler.data_min_[0]}")
            print(f"Max Volume : {self.scaler.data_max_[1]}, Min Volume: {self.scaler.data_min_[1]}")
            print("===== complete generating wbt =====")
            print("Num of Up trend:{}".format(list(t_data.reshape(-1)).count(0)))
            print("Num of Down trend:{}".format(list(t_data.reshape(-1)).count(1)))
            print("Num of Side trend:{}".format(list(t_data.reshape(-1)).count(2)))

        return wb_data, t_data


    def generate_image(self, w=5, v=4, info = True):
        """
        주가 수치 이미지화를 위한 메서드

        64 x 64 사이즈 이미지
        (1) single column represents a single day
        (2) The top part of the matrix represents the relative value of the closing price
        (3) The lower part of the matrix represents the relative value of the volume
        (4) Two rows in the middle of the chart are empty (has zero value)
        (5) All price, volume data are min-max normalized for visualization
        """
        
        size = self.size
        bin_num = int((size-2) / 2)
        wb_data, t_data = self.__generate_wbt(w=w, v=v, info = info)

        upper_data = np.zeros((wb_data.shape[0], bin_num, wb_data.shape[1]))
        lower_data = np.zeros((wb_data.shape[0], bin_num, wb_data.shape[1]))
        middle_data = np.zeros((wb_data.shape[0], 2, wb_data.shape[1]))

        for i in range(wb_data.shape[0]):

            price_values = wb_data[i, : ,0].astype(np.int8)
            volume_values = wb_data[i, :, 1].astype(np.int8)

            price_encoded = np.zeros((price_values.size, bin_num))
            price_encoded[np.arange(price_values.size), price_values] = 1
            price_encoded = np.flip(price_encoded, axis=1)
            price_encoded = np.transpose(price_encoded)

            volume_encoded = np.zeros((volume_values.size, bin_num))
            volume_encoded[np.arange(volume_values.size), volume_values] = 1
            volume_encoded = np.flip(volume_encoded, axis=1)
            volume_encoded = np.transpose(volume_encoded)

            upper_data[i] = price_encoded
            lower_data[i] = volume_encoded

        image_data = np.concatenate([upper_data, middle_data, lower_data], axis=1)
        image_data = image_data.reshape(-1, 1, size, size)
        label_data = F.one_hot(torch.LongTensor(t_data).view(-1), num_classes=3) if v!=0 else None
        label_data = np.array(label_data)
        if info:
            print("===== complete generating image =====")
            print("Image shape:{}".format(image_data.shape))
            print("Label shape:{}".format(label_data.shape))
            
        return image_data, label_data


class CustomDataset(Dataset):
    def __init__(self, x_all:np.array, y_all:np.array):
        self.x_data = x_all
        self.y_data = y_all
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


class TrainingModel:
    def __init__(self, model, x_all, y_all):
        self.model = model
        self.x_all = x_all
        self.y_all = y_all
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x_all, self.y_all, test_size=0.2, random_state=0)

    def __get_dataloader(self, batch_size, shuffle):
        dataset = CustomDataset(self.x_train, self.y_train)
        return DataLoader(dataset, batch_size, shuffle)

    def train(self, epochs, batch_size=32, shuffle=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        dataloader = self.__get_dataloader(batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epochs + 1):
            running_loss = 0.0
            for batch_idx, samples in enumerate(dataloader):
                x_train, y_train = samples
                prediction = self.model(x_train)
                loss = self.criterion(prediction, y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if batch_idx % 30 == 29:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 30:.3f}')
                    running_loss = 0.0

    def test(self):
        self.model.eval()
        x_test = torch.FloatTensor(self.x_test)
        y_test = torch.FloatTensor(self.y_test)

        test_pred = self.model(x_test)
        _, test_pred = torch.max(test_pred, 1)
        _, test_target = torch.max(y_test, 1)
        test_pred = np.array(test_pred.view(-1))
        test_target = np.array(test_target.view(-1))
        c = list(test_pred == test_target)
        o = c.count(1)
        x = c.count(0)
        score = (o / len(c)) * 100
        print(score)
        self.model.train()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        
        

class ResNet(nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.resnet = resnet18(pretrained=True)
        self.start = nn.Conv2d(1, 3, kernel_size=3, stride=1)
        self.layer1 = nn.Linear(1000, 512)
        self.layer2 = nn.Linear(512, 64)
        self.layer3 = nn.Linear(64, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.start(x)
        x = self.resnet(x)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x

    def predict(self, series_data:pd.DataFrame):
        self.eval()
        self.gd = GenerateDataset(data=series_data, size=self.image_size)
        image, _ = self.gd.generate_image(v=0, info=False)
        image = torch.FloatTensor(image)
        pred = self(image)
        _, pred = torch.max(pred, 1)
        self.train()
        return pred



