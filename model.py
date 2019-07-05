# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityBlock(nn.Module):
  """
  Implementing an identity block where the dimensions of both the opearands 
  while implementing the skip connection should be same.
  
  Inputs:
  X: Input tensor of shape(m, n_channels, n_H, n_W)
  f: filter size for the middle convolutional layer of the main path.
  filters: A list of 3 integers specifying the number of filters at each layer
           in the main path.
  in_channels = Number of input channels
  """
  
  def __init__(self, f, filters, in_channels):
    super(IdentityBlock, self).__init__()
    
    F1, F2, F3 = filters
    
    # First component in the main path
    self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=(1,1), stride=(1,1))
    self.batch_norm1 = nn.BatchNorm2d(F1)
    
    # Second component in the main path
    # Since PyTorch does not implement valid padding and same padding,
    # manually apply padding to cover case of both even and odd filter size.
    pad = (f-1)//2
    self.conv2 = None
    self.is_zero_pad = False
    if f%2 == 0: # If the filter size is even, padding only the right side.
      limit = f-1
      pad_1 = (f-1)//2
      pad_2 = limit-pad_1
      self.is_zero_pad = True
      self.zero_pad = nn.ZeroPad2d((pad_1, pad_2, pad_1, pad_2))
      self.conv2 = nn.Conv2d(F1, F2, kernel_size=(f,f), stride=(1,1), padding=0)
    else:
      self.conv2 = nn.Conv2d(F1, F2, kernel_size=(f,f), stride=(1,1), 
                             padding=pad)
    self.batch_norm2 = nn.BatchNorm2d(F2)
    
    # Third component in the main path
    self.conv3 = nn.Conv2d(F2, F3, kernel_size=(1,1), stride=(1,1))
    self.batch_norm3 = nn.BatchNorm2d(F3)
    
    self.relu = nn.ReLU()
    
  def layer(self, X):
    X_shortcut = X
    
    X = self.relu(self.batch_norm1(self.conv1(X)))

    if self.is_zero_pad:
      X = self.relu(self.batch_norm2(self.conv2(self.zero_pad(X))))
    else:
      X = self.relu(self.batch_norm2(self.conv2(X)))

    X = self.batch_norm3(self.conv3(X))
    
    X += X_shortcut
    return self.relu(X)


class ConvolutionalBlock(nn.Module):
  """
  Implementing the convolutional block where the operands of the skip
  connection are of different sizes.
  """
  
  def __init__(self, f, filters, in_channels, stride=2):
    super(ConvolutionalBlock, self).__init__() 
    
    # Retrieving the filters
    F1, F2, F3 = filters
     
    # First component in main path
    self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=(1,1), 
                           stride=(stride, stride))
    self.batch_norm1 = nn.BatchNorm2d(F1)
    
    # Second component in the main path
    # Since PyTorch does not implement valid padding and same padding,
    # manually apply padding to cover case of both even and odd filter size.
    pad = (f-1)//2
    self.conv2 = None
    self.is_zero_pad = False
    if f%2 == 0: # If the filter size is even, padding only the right side.
      limit = f-1
      pad_1 = (f-1)//2
      pad_2 = limit-pad_1
      self.is_zero_pad = True
      self.zero_pad = nn.ZeroPad2d((pad_1, pad_2, pad_1, pad_2))
      self.conv2 = nn.Conv2d(F1, F2, kernel_size=(f,f), stride=(1,1), padding=0)
    else:
      self.conv2 = nn.Conv2d(F1, F2, kernel_size=(f,f), stride=(1,1),
                             padding=pad)
    self.batch_norm2 = nn.BatchNorm2d(F2)
    
    # Third component in the main path
    self.conv3 = nn.Conv2d(F2, F3, kernel_size=(1,1), stride=(1,1))
    self.batch_norm3 = nn.BatchNorm2d(F3)
    
    # Shortcut path. Making sure the dimensions match.
    self.conv_shortcut = nn.Conv2d(in_channels, F3, kernel_size=(1,1), 
                                   stride=(stride, stride))
    # this is same as the third batch norm. Seperate for readability purpose.
    self.batch_norm_shortcut = nn.BatchNorm2d(F3) 
    
    self.relu = nn.ReLU()
    
  def layer(self, X):
    X_shortcut = X
    X = self.relu(self.batch_norm1(self.conv1(X)))
    
    if self.is_zero_pad:
      X = self.relu(self.batch_norm2(self.conv2(self.zero_pad(X))))
    else:
      X = self.relu(self.batch_norm2(self.conv2(X)))
    
    X = self.batch_norm3(self.conv3(X))
    
    # Implement the shortcut path
    X_shortcut = self.batch_norm_shortcut(self.conv_shortcut(X_shortcut))
    X += X_shortcut
      
    return self.relu(X)

class Resnet(nn.Module):
  """
  
  This takes in the images shape and the number of output classes.
  The images shape must be a tuple of the following format: (m, n_c, height, width)
  where M -> batch size
        n_c -> number of channels
        height -> image height in pixels
        width -> image width in pixels
  
  The input for the images will be: (m, num_channels, height, width)
  Output of pooling = ((i-f)/s) + 1
  """
  def __init__(self,in_shape, num_classes):
    super(Resnet, self).__init__()
    
    if len(in_shape) == 4:
      in_channels = in_shape[1]
    else:
      in_channels = in_shape[0]
    
    self.relu = nn.ReLU()
    
    self.initial_zero_pad = nn.ZeroPad2d((3,3)) # Output= (32,32)
    
    # Stage 1
    self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=(7,7),
                          stride=(2,2), padding=0) # Output -> (16,16,64)
    self.batch_norm1 = nn.BatchNorm2d(64)
    self.max_pool_1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)) # Output->(7,7)
    # Output channels = 64
    
    # Stage 2
    self.conv_block_2_1 = ConvolutionalBlock(f=3, filters=[64, 64, 256], in_channels=64, stride=2) # Output -> (4, 4, 256)
    self.id_block_2_1 = IdentityBlock(f=3, filters=[64, 64, 256], in_channels=256) # Output -> (4, 4, 256)
    self.id_block_2_2 = IdentityBlock(f=3, filters=[64, 64, 256], in_channels=256) # Output -> (4, 4, 256)
    # Output channels = 256
    
    # Stage 3
    self.conv_block_3_1 = ConvolutionalBlock(f=3, filters=[128,128,512], in_channels=256, stride=2) # Output -> (512, 2, 2)
    self.id_block_3_1 = IdentityBlock(f=3, filters=[128,128,512], in_channels=512) # Output -> (512, 2, 2)
    self.id_block_3_2 = IdentityBlock(f=3, filters=[128,128,512], in_channels=512) # Output -> (512, 2, 2)
    self.id_block_3_3 = IdentityBlock(f=3, filters=[128,128,512], in_channels=512) # Output -> (512, 2, 2)
    # Output channels = 512
    
    # Stage 4
    self.conv_block_4_1 = ConvolutionalBlock(f=3, filters=[256, 256, 1024], in_channels=512, stride=2) # Output -> (1024, 1, 1)
    self.id_block_4_1 = IdentityBlock(f=3, filters=[256, 256, 1024], in_channels=1024) # Output -> (1024, 1, 1)
    self.id_block_4_2 = IdentityBlock(f=3, filters=[256, 256, 1024], in_channels=1024) # Output -> (1024, 1, 1)
    self.id_block_4_3 = IdentityBlock(f=3, filters=[256, 256, 1024], in_channels=1024) # Output -> (1024, 1, 1)
    self.id_block_4_4 = IdentityBlock(f=3, filters=[256, 256, 1024], in_channels=1024) # Output -> (1024, 1, 1)
    self.id_block_4_5 = IdentityBlock(f=3, filters=[256, 256, 1024], in_channels=1024) # Output -> (1024, 1, 1)
    # Output channels = 1024
    
    # Stage 5
    self.conv_block_5_1 = ConvolutionalBlock(f=3, filters=[512, 512, 2048], in_channels=1024, stride=2) # Output -> (2048, 1, 1)
    self.id_block_5_1 = IdentityBlock(f=3, filters=[512, 512, 2048], in_channels=2048) # Output -> (2048, 1, 1)
    self.id_block_5_2 = IdentityBlock(f=3, filters=[512, 512, 2048], in_channels=2048) # Output -> (2048, 1, 1)
    
    self.avg_pool = nn.AvgPool2d(kernel_size=(2,2), padding=(0, 0))
    
    # Fully connected layers after flattening
    self.fc1 = nn.Linear(2048, 512)
    self.fc2 = nn.Linear(512, 64)
    self.fc3 = nn.Linear(64, 10)
    self.dropout = nn.Dropout(p=0.3)
    
  def forward(self, X):
    
    X = self.initial_zero_pad(X)
    
    X = self.conv1(X)
    X = self.batch_norm1(X)
    X = self.max_pool_1(X)
    
    X = self.conv_block_2_1.layer(X)
    X = self.id_block_2_1.layer(X)
    X = self.id_block_2_2.layer(X)
    
    X = self.conv_block_3_1.layer(X)
    X = self.id_block_3_1.layer(X)
    X = self.id_block_3_2.layer(X)
    X = self.id_block_3_3.layer(X)
    
    X = self.conv_block_4_1.layer(X)
    X = self.id_block_4_1.layer(X)
    X = self.id_block_4_2.layer(X)
    X = self.id_block_4_3.layer(X)
    X = self.id_block_4_4.layer(X)
    X = self.id_block_4_5.layer(X)
    
    X = self.conv_block_5_1.layer(X)
    X = self.id_block_5_1.layer(X)
    X = self.id_block_5_2.layer(X)
    
    X = self.avg_pool(X)
    
    
    X = X.view(X.shape[0], -1)
    
    X = self.dropout(F.relu(self.fc1(X)))
    X = self.dropout(F.relu(self.fc2(X)))
    
    return self.fc3(X)
    
    
# resnet = Resnet((64, 3, 32,32),5)
# image = torch.randn((64, 3, 32,32))
# output = resnet(image)
# print(output.shape)

def get_model():
  resnet = Resnet((64, 1, 32, 32),5)
  checkpoint = torch.load('check.pth', map_location=lambda storage, loc: storage)
  resnet.load_state_dict(checkpoint)
  resnet.eval()
  
  return resnet