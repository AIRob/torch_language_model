import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LanguageModel(nn.Module):

    def __init__(self):
        super(LanguageModel, self).__init__()

        # Set the embedding layer
        self.embedding = nn.Embedding(self.config['n_words'], 300)

        # Initialize the embedding weights
        weights = np.load(self.config['initial_embed_weights'])

        # embedding.weight is of size N_Embeddings x Embeddings_dim
        self.embedding.weight = weights

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

    def sentence_forward(self, x):
        x = self.embedding(x)
        return x

    def forward(self, x):
        # Separate `x` into 3 distinct sentences
        target_word, synonym, antonym = x

        # Run the rest of the network in each one of them
        target_word = self.sentence_forward(target_word)
        synonym = self.sentence_forward(synonym)
        antonym = self.sentence_forward(antonym)

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
        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features
