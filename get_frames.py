# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:08:33 2024

@author: chels
"""

import glob
import torch
import os
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
from utils import vid_to_frames, transform_frames, save_frames


mainpath = r'C:\Users\chels\.keras\datasets\costar_block_stacking_dataset_v0.4\johns_hopkins_costar_dataset\blocks_only'
datapath = fr'{mainpath}\*.h5f'
to_savepath = fr'{mainpath}\frames'
from_savepath = fr'{mainpath}\frames\*.jpg'

vid_files_paths = glob.glob(datapath)
#vid_files_list = list(map(os.path.basename, vid_files_paths))

def make_frames(listof_vid_files_paths, save_path):
    idx = []
    for file_path in listof_vid_files_paths:
        images, labels, labelnames = vid_to_frames(file_path)
        if images is not None:
            images = transform_frames(images)
            #filename = os.path.splitext(os.path.basename(file_path))[0]
            #filename = Path(file_path).stem
            path = Path(file_path)
            filename = path.name.removesuffix("".join(path.suffixes))
            save_frames(images, labels, filename, save_path)
            idx.append(filename)
    return

def make_frames2(listof_vid_files_paths, save_path):
    idx = []
    for file_path in listof_vid_files_paths:
        images, labels, labelnames = vid_to_frames(file_path)
        if images is not None:
            #images = transform_frames(images)
            #filename = os.path.splitext(os.path.basename(file_path))[0]
            #filename = Path(file_path).stem
            path = Path(file_path)
            filename = path.name.removesuffix("".join(path.suffixes))
            save_frames(images, labels, filename, save_path)
            idx.append(filename)
    return

def get_frames(save_path):
    vid_dict = {}
    save_paths = glob.glob(save_path)
    for file_path in save_paths:
        img_file_name = os.path.splitext(os.path.basename(file_path))[0]
        vid, framenum, labelnum = img_file_name.rsplit("_", 2)
        _, frame = mysplit(framenum)
        _, label = mysplit(labelnum)
        
        jpg_image = plt.imread(file_path)
        image = transforms.ToTensor()(jpg_image)
        
        if vid in vid_dict:
            vid_dict[vid].append((image, label))
        else:
            vid_dict[vid] = [(image, label)]
        
    frames, labels = extract_frames_and_labels(vid_dict)
    #each item in frames should have shape (5,3,224,224)
    
    return frames, labels

def get_frames2(save_path):
    vid_dict = {}
    save_paths = glob.glob(save_path)
    for file_path in save_paths:
        img_file_name = os.path.splitext(os.path.basename(file_path))[0]
        vid, framenum, labelnum = img_file_name.rsplit("_", 2)
        _, frame = mysplit(framenum)
        _, label = mysplit(labelnum)
        
        jpg_image = plt.imread(file_path)
        image = transforms.ToTensor()(jpg_image)
        
        if vid in vid_dict:
            vid_dict[vid].append((image, label))
        else:
            vid_dict[vid] = [(image, label)]
        
    frames, labels = extract_frames_and_labels(vid_dict)
    images = transform_frames(frames)
    #each item in frames should have shape (5,3,224,224)
    
    return images, labels

def mysplit(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    tail = float(tail)
    return head, tail

def extract_frames_and_labels(dictionary):
    list_of_frames = []
    list_of_labels = []

    for key, value in dictionary.items():
        frames = []
        labels = []
        for frame, label in value:
            frames.append(frame)
            labels.append(label)
        frames_tensor = torch.stack(frames)
        labels_array = np.array(labels)
        list_of_frames.append(frames_tensor)
        list_of_labels.append(labels_array)

    return list_of_frames, list_of_labels

#make_frames(vid_files_paths, to_savepath)
#get_frames(from_savepath)
# v = r'\2018-05-08-17-26-53_example000016.success.h5f'
# test_v = mainpath + v
# test_sv = r'C:\Users\chels\.keras\datasets\costar_block_stacking_dataset_v0.4\johns_hopkins_costar_dataset\blocks_only\test'
# from_test_sv = fr'{test_sv}\*.jpg'
# make_frames2([test_v], test_sv)
# get_frames2(from_test_sv)

