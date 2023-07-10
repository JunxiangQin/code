import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy import stats

class OmicDataset(Dataset):
    def __init__(self,root,file):
        raw_data = pd.read_csv(root + file)
        # print(raw_data)
        patients_df = raw_data.drop_duplicates(['case_id']) # drop repeated cases, each case has several images

        # input data
        cnv = patients_df[patients_df.columns[patients_df.columns.str.contains('_cnv')]]
        mut = []
        rna = patients_df[patients_df.columns[patients_df.columns.str.contains('_rnaseq')]]
        columns = raw_data.colunms
        for column in columns[10:]:
            if '_cnv' in column or '_rnaseq' in column:
                pass
            else:
                mut.append(patients_df[column])

        cnv = np.array(cnv)
        rna = np.array(rna)
        mut = np.array(mut).T
        molecule = np.concatenate((cnv, mut, rna), axis=1)
        ground_label = patients_df['disc_label'].to_numpy()
        censor = patients_df['censorship'].to_numpy()
        event_time = patients_df['survival_months'].to_numpy()

        self.molecule = molecule
        self.censor = censor
        self.event_time = event_time
        self.ground_label = ground_label

    def __getitem__(self,item):
         molecule, censor, ground_label = self.molecule[item], self.censor[item], self.ground_label[item]
         event_time = self.event_time[item]
         return molecule, censor, ground_label, event_time

    def __len__(self):
        return len(self.ground_label)

    def featue_dim(self):
        return self.molecule.shape[1]

class WsiDataset(Dataset):
    def __init__(self,root, file, wsi_path, label_col = 'survival_months', n_bins = 4, eps = 1e-6 ):
        raw_data = pd.read_csv(root + file) # ground_truth file
        pt_data = os.list(wsi_path) # wsi.pt file
        pt_data.sort()

        uncensored_df = raw_data[raw_data['censorship'] < 1]  # censorship=1:右删失 censorship=0:未删失
        _, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = raw_data[label_col].max() + eps
        q_bins[0] = raw_data[label_col].min() - eps
        disc_labels, q_bins = pd.cut(raw_data[label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        ground_label = disc_labels.values.astype(int)

        ground_label = disc_labels.values.astype(int)
        # print('ground_label is ', ground_label.shape)

        # 保存每个case的censorship，用于计算loss
        censor = np.array(raw_data['censorship'])
        # print('censorship   is ', censor.shape)

        event_time = np.array(raw_data[label_col])
        # print('event_time   is ', event_time.shape)

        self.path = wsi_path
        self.pt_data = pt_data
        self.ground_label = ground_label
        self.censor = censor
        self.event_time = event_time

    def __getitem__(self, item):
         event_time, censor, ground_label = self.event_time[item], self.censor[item], self.ground_label[item]
         wsi = torch.load(self.path + self.pt_data[item])
         return wsi, censor, ground_label, event_time

    def __len__(self):
        return len(self.pt_data)






