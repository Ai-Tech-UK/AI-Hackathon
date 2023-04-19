#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

data_path = r'E:\AI_Tech_UK\AI-LABS\raw-img'
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

label_list = []
for root, dirs, files in os.walk(data_path):
    for name in files:
        label = os.path.basename(root)
        if label not in label_list:
            label_list.append(label)
            
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
labels = [label_list.index(os.path.basename(os.path.dirname(x[0]))) for x in dataset.imgs]
dataset.targets = labels

num_samples = len(dataset)
indices = list(range(num_samples))
split1, split2 = int(num_samples*0.7), int(num_samples*0.85)
train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


# In[19]:


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


# In[20]:


import torch.optim as optim

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10): 
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')


# In[50]:


translate = {
    "cane": "dog",
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
    "squirrel": "scoiattolo"
}

from PIL import Image
test_image = Image.open(r'E:\AI_Tech_UK\AI-LABS\test7.jpg')
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


# In[47]:


# Save the trained model
torch.save(net.state_dict(), r'E:\AI_Tech_UK\AI-LABS\aitech-model.pth')



# In[ ]:





# In[ ]:




