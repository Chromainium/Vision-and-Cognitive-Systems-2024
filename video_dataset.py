# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:37:32 2024

@author: chels
"""

from torch.utils.data import Dataset
from get_frames import get_frames
#import torch

class VideoDataset(Dataset):
    def __init__(self, vid_files_paths):
        self.frames, self.labels = get_frames(vid_files_paths)
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frames = self.frames[idx]
        labels = self.labels[idx]
        #labelnames = self.labelnames[idx]
        return frames, labels#, labelnames
