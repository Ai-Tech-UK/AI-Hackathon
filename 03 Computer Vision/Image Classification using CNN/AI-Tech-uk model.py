#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Define the label list
label_list = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'scoiattolo']

# Define a dictionary to translate the labels to their corresponding classes
translate = {"cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "dog": "cane",
    "cavallo": "horse",
    "elephant" : "elefante",
    "butterfly": "farfalla",
    "chicken": "gallina",
    "cat": "gatto",
    "cow": "mucca",
    "spider": "ragno",
    "squirrel": "scoiattolo"}


net = Net() # create a new instance of the model
net.load_state_dict(torch.load(r'E:\AI_Tech_UK\AI-LABS\ai-tech.pt'))

from PIL import Image
test_image = Image.open(r'E:\AI_Tech_UK\AI-LABS\test3.jpeg')
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_image = test_transform(test_image)
test_image = test_image.unsqueeze(0)

with torch.no_grad():
    outputs = net(test_image)
    _, predicted = torch.max(outputs, 1)
    predicted_label = label_list[predicted.item()]
    if predicted_label in translate:
        translated_label = translate[predicted_label]
        print('Its a:', translated_label)
    else:
        print('Its a:', predicted_label)


# In[ ]:





# In[ ]:




