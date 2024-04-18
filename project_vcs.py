# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:10:28 2024

@author: chels
"""

from video_dataset import VideoDataset
from torch.utils.data import DataLoader
from resnet18_rnn import Resnet18Rnn
import torch.nn as nn
import torch.optim as optim
import utils


mainpath = r'C:\Users\chels\.keras\datasets\costar_block_stacking_dataset_v0.4\johns_hopkins_costar_dataset\blocks_only'
datapath = fr'{mainpath}\*.h5f'
savepath = fr'{mainpath}\frames'
to_savepath = fr'{mainpath}\frames'
from_savepath = fr'{mainpath}\frames\*.jpg'

#img_files_paths = glob.glob(from_savepath)

video_dataset = VideoDataset(from_savepath)

# Define batch size
batch_size = 5

# Create a DataLoader for the dataset
data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# Assuming you have defined the necessary parameters
num_classes = 41  # But theres technically 41 classes (0-40)
hidden_size = 256  # 205 = 41*5 Size of the hidden state in the RNN decoder

# Create an instance of the Resnet18Rnn model
model = Resnet18Rnn(num_classes=num_classes, hidden_size=hidden_size)

# Print the model architecture
# print(model)

# Define loss criterion and optimizer
criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss accepts ground truth labels directly as integers in [0, N_CLASSES
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

trained_model, loss_hist = utils.train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs=50)

utils.plot_loss_history(loss_hist)

import torch

from get_frames import get_frames

test1_folder = r"C:\Users\chels\.keras\datasets\costar_block_stacking_dataset_v0.4\johns_hopkins_costar_dataset\blocks_only\test"
from_test1_folder = fr'{test1_folder}\*.jpg'

frames, labels = get_frames(from_test1_folder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testset = VideoDataset(from_test1_folder)
testdl = DataLoader(testset)

trained_model.eval()
trained_model.to(device)

f = frames[0]
f = f.unsqueeze(0)

with torch.no_grad():
    out = trained_model(f.to(device)).cpu()
    print(out.shape)
    print(out)
    #pred = torch.argmax(out).item()
    #print(pred)

print(labels)