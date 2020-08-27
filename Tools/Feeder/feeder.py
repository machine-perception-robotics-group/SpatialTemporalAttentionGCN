# -*- coding: utf-8 -*-
import numpy as np
import pickle
import torch
import torch.utils.data

from . import feeder_tools


class Feeder(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 frame_size,
                 normalization,
                 random_shift,
                 valid_choose,
                 frame_thinning,
                 random_choose,
                 repeat_padding,
                 random_move,
                 add_noise,
                 frame_normalization):
        self.data_path = data_path
        self.label_path = label_path
        self.frame_size = frame_size
        self.normalization = normalization
        self.random_shift = random_shift
        self.valid_choose = valid_choose
        self.frame_thinning = frame_thinning
        self.random_choose = random_choose
        self.repeat_padding = repeat_padding
        self.random_move = random_move
        self.add_noise = add_noise
        self.frame_normalization = frame_normalization


        self.load_data()

        if self.normalization:
            self.get_mean_map()

    def load_data(self):
        # load label
        if '.pkl' in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            raise ValueError

        # load data
        self.data = np.load(self.data_path)
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data = np.array(self.data[index])
        label = self.label[index]
        name = self.sample_name[index]

        if self.normalization:
            data = (data - self.mean_map) / self.std_map
        if self.random_shift:
            data = feeder_tools.random_shift(data)
        if self.valid_choose:
            data = feeder_tools.valid_choose(data, self.frame_size)
        elif self.frame_thinning:
            data = feeder_tools.frame_thinning(data, self.frame_size)
        elif self.random_choose:
            data = feeder_tools.random_choose(data, self.frame_size)
        elif self.frame_size > 0:
            data = feeder_tools.auto_pading(data, self.frame_size)
        if self.repeat_padding:
            data = feeder_tools.repeat_padding(data)
        if self.random_move:
            data = feeder_tools.random_move(data)
        if self.add_noise:
            data = feeder_tools.add_noise(data)
        if self.frame_normalization:
            data = feeder_tools.frame_normalization(data)

        return data, label, name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
