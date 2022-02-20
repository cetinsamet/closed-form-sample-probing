# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# meta_dset.py
#
# Written by cetinsamet -*- cetin.samet@metu.edu.tr
# September, 2021
# --------------------------------------------------
from torch.utils.data import Dataset
import numpy as np
import torch

FN = torch.from_numpy


class MetaDataset(Dataset):
    def __init__(self, data, opt):
        super(MetaDataset, self).__init__()
        
        self.n_task = opt.n_task
        self.n_probe_train = opt.n_probe_train
        self.n_probe_val = opt.n_probe_val
        self.k_probe_train = opt.k_probe_train
        self.k_probe_val = opt.k_probe_val
        self.n_subset = opt.n_subset
        
        self.train_feature = data.train_feature
        self.train_label = data.train_label

        self.train_feature = self.train_feature.numpy()
        self.train_label = self.train_label.numpy()
        self.seenclasses = np.unique(self.train_label)
        
        self.set_tasks() # init tasks
        
    def set_tasks(self):
        
        X_SUPPORT, Y_SUPPORT = [], []
        X_QUERY, Y_QUERY = [], []
        
        for _ in range(self.n_task):
            
            task_classes = self.seenclasses[np.random.permutation(len(self.seenclasses))]
            support_classes = task_classes[:self.n_probe_train]
            possible_query_classes = task_classes[self.n_probe_train:]
            
            ##### support
            x_support, y_support = [], []
            for c in support_classes:
                c_indices = (self.train_label == c)
                c_k_indices = np.random.permutation(np.sum(c_indices))[:self.k_probe_train]

                x_support.append(self.train_feature[c_indices][c_k_indices])
                y_support.append(self.train_label[c_indices][c_k_indices])
                
            # concatenate
            x_support, y_support = np.concatenate(x_support), np.concatenate(y_support)
            
            # shuffle and store
            random_indices = np.random.permutation(x_support.shape[0])
            X_SUPPORT.append(x_support[random_indices])
            Y_SUPPORT.append(y_support[random_indices])
            # delete
            del x_support; del y_support
            
            ##### query
            x_query, y_query = [], []           
            for _ in range(self.n_subset):
                query_classes = possible_query_classes[np.random.permutation(len(possible_query_classes))][:self.n_probe_val]
                
                x_subset_query, y_subset_query = [], []
                for c in query_classes:
                    c_indices = (self.train_label == c)
                    c_k_indices = np.random.permutation(np.sum(c_indices))[:self.k_probe_val]   
                    x_subset_query.append(self.train_feature[c_indices][c_k_indices])
                    y_subset_query.append(self.train_label[c_indices][c_k_indices])
                    
                # concatenate
                x_subset_query = np.concatenate(x_subset_query)
                y_subset_query = np.concatenate(y_subset_query)
                # shuffle and store
                random_indices = np.random.permutation(x_subset_query.shape[0])
                x_query.append(x_subset_query[random_indices])
                y_query.append(y_subset_query[random_indices])
                # delete
                del x_subset_query; del y_subset_query
                
            x_temp_query, y_temp_query = [], []
            for i in range(self.n_subset):
                # shuffle and store
                random_indices = np.random.permutation(x_query[i].shape[0])
                x_temp_query.append(x_query[i][random_indices])
                y_temp_query.append(y_query[i][random_indices])
            
            X_QUERY.append(x_temp_query)
            Y_QUERY.append(y_temp_query)

            # delete
            del x_temp_query; del y_temp_query
            
        self.tasks = (X_SUPPORT, Y_SUPPORT, X_QUERY, Y_QUERY)
        
    def __getitem__(self, index):
        return self.tasks[0][index], self.tasks[1][index], self.tasks[2][index], self.tasks[3][index]
    
    def __len__(self):
        return self.n_task
    