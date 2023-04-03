from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
#import calMetric as m
from dataset import MyDataset
import calData as d
import os
from classifier import SentimentClassifier
import pickle

def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):
    net1 = SentimentClassifier()
    net1.load_state_dict(torch.load(f"./models/lstm.pth", map_location=torch.device('cpu')))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net1.parameters(),lr= 0.001)

    print('Loading files ...')
    test_files = get_file("./data/in/test")
    
    print("Building vocabulary ...")
    vocab = load_vocab()
    
    print("Setting up dataloaders ...")
    test_dataset = MyDataset(test_files, vocab)
    test_dataloader_in = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True,collate_fn=MyDataset.collate_fun, num_workers=0)


    d.testData(net1, criterion, CUDA_DEVICE, test_dataloader_in, test_dataloader_in, nnName, dataName, epsilon, temperature) 
    # m.metric(nnName, dataName)

def get_file(dir):
  paths_list = [] 
  for par, dirs, files in os.walk(dir):
    if "pos" in par or "neg" in par:
      paths_list.extend([par + "/" + f for f in files])
  return paths_list

def load_vocab():
    with open("data/vocabulary.pkl", "rb") as file:
        vocab = pickle.load(file)
    return vocab