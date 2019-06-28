import numpy as np
import tensorflow as tf
import torch
from torch.utils.data.dataset import Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def dataLoader(trX, trY, batch_size):
    datalen = np.shape(trX)[0]
    seq = np.random.permutation(datalen)
    itersize = int(np.ceil(datalen/float(batch_size)))
    for idx in range(itersize):
        if idx < itersize -1 :
            yield (trX[idx*batch_size:(idx+1)*batch_size], trY[idx*batch_size:(idx+1)*batch_size])
        else:
            yield (trX[idx*batch_size:], trY[idx*batch_size:])

class tfMNIST2pytorchMNIST(Dataset):
    def __init__(self, tX, tY):
        self.X = tX
        self.Y = tY
    
    def __len__(self):
        return np.shape(self.X)[0]

    def __getitem__(self, idx):
        image = self.X[idx, :].reshape(28, 28)
        label = np.where(self.Y[idx, :] == 1.0 )[0].item()
        return image, label
