from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pickle

class SynonymsAntonyms(Dataset):
    def __init__(self, input_file, transform=None):
        self.word_triples = pickle.load(open(input_file, "rb"))
        self.transform = transform

        # Load vocabulary

    def __len__(self):
        return len(self.word_triples)

    def __getitem__(self, idx):
        sample = self.word_triples[idx]

        if self.transform:
            sample = self.transform(sample)

        # Instead of being a triple like
        #    ('good', 'goodness', 'badness')
        # I want to return a triple that looks like
        #    (101, 324, 671)
        # Where each of these numbers is the index of the words in the
        # vocabulary
        return sample
