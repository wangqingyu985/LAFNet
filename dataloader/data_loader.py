"""
Created by Mr. Qingyu Wang at 14:33 07.03.2025
E-mail address: 12013027@zju.edu.cn
"""
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


class PickleDatasetLoader(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tactile_data = []
        self.air_pressure = []
        self.adaptive_force = []

        for file in os.listdir(self.data_dir):
            with open(file=(self.data_dir + file), mode='rb') as f:
                data = pickle.load(f)
                self.tactile_data.append(data['tactile_data'])
                self.air_pressure.append(data['air_pressure'])
                self.adaptive_force.append(data['ground_truth'])

    def __len__(self):
        lists = [self.tactile_data, self.air_pressure, self.adaptive_force]
        if all(len(lst) == len(lists[0]) for lst in lists):
            return len(lists[0])
        else:
            raise ValueError("Data length and format are abnormal!")

    def __getitem__(self, idx):
        tactile_data = self.tactile_data[idx]
        air_pressure = self.air_pressure[idx]
        adaptive_force = self.adaptive_force[idx]
        return torch.tensor(tactile_data), torch.tensor(air_pressure), torch.tensor(adaptive_force)


if __name__ == '__main__':
    data_dir = '/media/wangqingyu/固态硬盘/ForceLearning/dataset/kiwi/'
    pickledatasetloader = PickleDatasetLoader(data_dir=data_dir)
    dataloader = DataLoader(pickledatasetloader, batch_size=1, shuffle=True)
    for tactile_data, air_pressure, adaptive_force in dataloader:
        print(tactile_data, air_pressure, adaptive_force)
        break
