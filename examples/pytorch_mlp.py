import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
from mnist_dataloader import dataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

# seed = np.random.randint(1000)
seed = 0
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def WeightInitializer(input_size, hidden_size, num_classes, initdev):
    w1 = initdev * np.transpose(np.random.randn(input_size, hidden_size)).astype(np.float32)
    w2 = initdev * np.transpose(np.random.randn(hidden_size, hidden_size)).astype(np.float32)
    wo = initdev * np.transpose(np.random.randn(hidden_size, num_classes)).astype(np.float32)
    return w1, w2, wo

def oneHot2num(x_label):
    return np.where(x_label == 1.0)[1]

# torch.manual_seed(seed)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# TF_train_dataset = tfMNIST2pytorchMNIST(trX, trY)
# TF_test_dataset = tfMNIST2pytorchMNIST(teX, teY)

# Hyper Parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 20
batch_size = 2**10
learning_rate = 0.001
initdev = 0.1

class MultilayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.fc3 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = MultilayerNN(input_size, hidden_size, num_classes)

# Weight initialization
# model.fc1.weight.data.normal_(0.0, initdev)
# model.fc2.weight.data.normal_(0.0, initdev)
# model.fc3.weight.data.normal_(0.0, initdev)

w1, w2, wo = WeightInitializer(input_size, hidden_size, num_classes, initdev)
model.fc1.weight.data = torch.from_numpy(w1)
model.fc2.weight.data = torch.from_numpy(w2)
model.fc3.weight.data = torch.from_numpy(wo)
print('w1:', w1)

# Use cuda
model.cuda()   
   
# Set loss function
criterion = nn.CrossEntropyLoss()  

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)


# Train the Model
for epoch in range(num_epochs):
    i = 0
    # for i, (images, labels) in enumerate(train_loader):  
    if epoch > 0:
        for i, (images, labels) in enumerate(dataLoader(trX, trY, batch_size)):
            # Convert torch tensor to Variable
            images = torch.from_numpy(images)
            labels = torch.from_numpy(oneHot2num(labels))
            images = Variable(images.view(-1, 28*28).cuda())
            labels = Variable(labels.cuda())
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # for images, labels in test_loader:
    total, correct = 0, 0
    for i, (images_test, labels_test) in enumerate(dataLoader(teX, teY, batch_size)):
        images_test = torch.from_numpy(images_test)
        labels_test = torch.from_numpy(oneHot2num(labels_test))
        images_test = Variable(images_test.view(-1, 28*28), requires_grad=False).cuda()
        outputs_test = model(images_test)
        _, predicted = torch.max(outputs_test.data, 1)
        total += labels_test.size(0)
        correct += (predicted.cpu() == labels_test).sum()

    print('Pytorch: Training Epoch  %d Test Accuracy: %.4f' %(epoch+1, correct /float(total))) 


save_path = './model.pkl'
torch.save(model.state_dict(),  save_path)
print("Model saved in path: %s" % save_path)  

