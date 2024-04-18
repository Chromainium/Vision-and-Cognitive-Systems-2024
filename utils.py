# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:41:02 2024

@author: chels
"""

import os
import torch
import h5py
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        model.train()
        #for inputs, labels, _ in dataloader:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            #labels = labels.to(torch.float32)

            optimizer.zero_grad()
            
            outputs = model(inputs) #should be (5,5,41)
            #raw_outputs = model(inputs)
            #print("outputs0 grad_fn attribute: ", outputs.grad_fn)
            
            #outputs = torch.argmax(raw_outputs, dim=2)
            #print("outputs1 grad_fn attribute: ", outputs.grad_fn)
            
            #print("OUTPUTS")
            #print(outputs)
            
            #print("LABELS")
            #print(labels)
            #print("\n")
            
            #outputs = outputs.to(torch.float64)
            #print("outputs2 grad_fn attribute: ", outputs.grad_fn)
            
            loss = criterion(outputs, labels)
            #Error labels.dtype = torch.int64 but outputs.dtype = torch.float32
            #RuntimeError: Expected floating point type for target with class probabilities, got Long

            #print("loss grad_fn attribute: ", loss.grad_fn)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        scheduler.step(running_loss)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        
        if num_epochs < 50:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        else:
            if epoch%10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model, loss_history

def plot_loss_history(loss_history):
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.show()

def save_model(model, optimizer, filepath):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)

def load_model(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def JpegToNumpy(jpeg):
    stream = io.BytesIO(jpeg)
    image = Image.open(stream)
    return np.asarray(image, dtype=np.uint8)


def ConvertImageListToNumpy(data):
    images = []
    for raw in data:
        img = JpegToNumpy(raw)
        images.append(img)
    return np.array(images, dtype=np.uint8)

def vid_to_frames(path, view_frames = False):
    #h5py_filename = '2018-05-15-14-13-32_example000008'
    #path = (fr'C:\Users\chels\.keras\datasets\costar_block_stacking_dataset_v0.4\johns_hopkins_costar_dataset\blocks_only\{h5py_filename}.success.h5f')
    try:
        with h5py.File(os.path.expanduser(path), 'r') as data:
            goal_frames = np.unique(data['gripper_action_goal_idx'])
            
            label_list = np.array(data['label'])[goal_frames]
            labelname_list = np.array(data['labels_to_name'])[label_list]
    
            # Get the images corresponding to the goals
            image_list = np.array(data['image'])[goal_frames]
    
            # The images are stored as JPEG files.
            # We now use functions defined above to convert the images to np arrays
            images = ConvertImageListToNumpy(image_list)
    
            # Turn the images into a tile
            tiled_images = np.squeeze(np.hstack(images))
    
            # Show the images with pyplot
            if view_frames:
                plt.imshow(tiled_images)
                plt.show()
            
            # for name in labelname_list:
            #     print(name)
            pass
        
    except OSError as e:
        print(f"Error opening file '{path}': {e}")
        return None, None, None
    
    # print('vid_to_frames output:', type(images))
    # print('vid_to_frames size:', images.shape)
    # print('\n')
    
    return images, label_list, labelname_list

import torchvision.transforms as transforms

def transform_frames(images, model_type="rnn"):
    #cropped_images = images[:,80:,:,:]
    # Pytorch requirements for Resnet inputs
    if model_type == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Pytorch requirements for other architectures
    # else:
    #     h, w = 112, 112
    #     mean = [0.43216, 0.394666, 0.37645]
    #     std = [0.22803, 0.22145, 0.216989]

    test_transformer = transforms.Compose([
                transforms.Resize((h,w)),
                #transforms.ToTensor(),
                transforms.Normalize(mean, std)]) 

    frames_tr = []
    for frame in images:
        #frame = Image.fromarray(frame)
        frame_tr = test_transformer(frame)
        frames_tr.append(frame_tr)
    
    #print('frames shape:', frames_tr[0].shape)
    #transformed_images = torch.stack(frames_tr)
    transformed_images = frames_tr
    #print('transformed:', transformed_images.shape)
    #print('\n')
    
    return transformed_images

def save_frames(images, label_list, h5py_filename, savepath):
    for idx, np_image in enumerate(images):
        #print(type(images))
        #print(images.size())
        #print(type(np_image))
        #print(np_image.size())
        #image = Image.fromarray(np_image)
        #os.mkdir()
        image = transforms.ToPILImage()(np_image)
        label = label_list[idx]
        image.save(fr"{savepath}/{h5py_filename}_image{idx}_label{label}.jpg", 'JPEG')