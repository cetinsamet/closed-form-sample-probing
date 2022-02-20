# -*- coding: utf-8 -*-
# --------------------------------------------------
#
# util.py
#
# Adapted from https://github.com/akshitac8/tfvaegan
# --------------------------------------------------
from sklearn import preprocessing
import scipy.io as sio
import numpy as np
import torch
import os

join = os.path.join


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        # ---------------------------------
        # Load image features and labels 
        if opt.finetune:
            data_embedding = sio.loadmat(join(opt.dataroot, opt.dataset, opt.image_embedding + '_finetuned.mat'))
        else: 
            data_embedding = sio.loadmat(join(opt.dataroot, opt.dataset, opt.image_embedding + '.mat'))
        feature = data_embedding['features'].T
        label = data_embedding['labels'].astype(int).squeeze() - 1
        # ---------------------------------
        # Load data splits
        data_splits = sio.loadmat(join(opt.dataroot, opt.dataset, opt.class_embedding + '_splits.mat'))
        trainval_loc = data_splits['trainval_loc'].squeeze() - 1
        train_loc = data_splits['train_loc'].squeeze() - 1
        val_unseen_loc = data_splits['val_loc'].squeeze() - 1
        test_seen_loc = data_splits['test_seen_loc'].squeeze() - 1
        test_unseen_loc = data_splits['test_unseen_loc'].squeeze() - 1    
        # ---------------------------------
        # Load attributes
        self.attribute = torch.from_numpy(data_splits['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0), self.attribute.size(1)) # normalize
        # ---------------------------------

        # --------------------------------------------------------------------------------------------
        # VALIDATION MODE
        if opt.validation:
            # train and val (test seen) samples are mixed
            _trainval_feature = feature[train_loc]
            _trainval_label = label[train_loc]
            n_sample = _trainval_feature.shape[0]
            n_train_sample = int(0.8 * n_sample)
            np.random.seed(16)
            order = np.random.permutation(n_sample)
            # train
            _train_feature = _trainval_feature[order[:n_train_sample]]
            _train_label = _trainval_label[order[:n_train_sample]]
            # test_seen
            _test_seen_feature = _trainval_feature[order[n_train_sample:]]
            _test_seen_label = _trainval_label[order[n_train_sample:]]

            # test unseen
            _test_unseen_feature = feature[val_unseen_loc]
            _test_unseen_label = label[val_unseen_loc]

            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(_train_feature)
                _test_seen_feature = scaler.transform(_test_seen_feature)
                _test_unseen_feature = scaler.transform(_test_unseen_feature)

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(_train_label).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(_test_unseen_label).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(_test_seen_label).long()
            else:
                self.train_feature = torch.from_numpy(_train_feature).float()
                self.train_label = torch.from_numpy(_train_label).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_label = torch.from_numpy(_test_unseen_label).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_label = torch.from_numpy(_test_seen_label).long()  
        # --------------------------------------------------------------------------------------------
        ### TEST MODE
        else:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        # --------------------------------------------------------------------------------------------
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.d_ft = self.train_feature.size(1)
        self.d_attr = self.attribute.size(1)

        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att
        