import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LanguageModel(nn.Module):

    def __init__(self, w2i, i2w, w2v):
        super(LanguageModel, self).__init__()
        self.w2i = w2i
        self.i2w = i2w

        # Set the embedding layer
        self.embedding = nn.Embedding(len(w2i), 300)

        # `embedding.weight` is of size N_Embeddings x Embeddings_dim
        self.embedding.weight = nn.Parameter(torch.from_numpy(w2v))

        # Freezes the weights of the embedding layer
        #self.embedding.weight.requires_grad = False

        # self.fc1 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)

    def predict(self, w_batched):
        return self.forward_once(w_batched)

    def forward_once(self, x):
        return self.embedding(x)

    def forward(self, x):
        # Notice that each of these elements is probably batched
        target_word, synonym, antonym = x

        # Run the rest of the network in each one of them
        target_word = self.forward_once(target_word)
        synonym = self.forward_once(synonym)
        antonym = self.forward_once(antonym)

        # Probably something more complicated...

        # Mix them all together back again
        return (target_word, synonym, antonym)

        # # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

