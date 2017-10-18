import argparse
import pickle
import os

import torch
from torch.utils.data import Dataset, DataLoader

from synspace import split_triples
from synspace.vocabulary_utils import load_vocabulary


class ToTensor():
    """Convert tuples in sample to Tensors."""

    def __call__(self, sample):
        # return {'target_word': torch.Tensor([sample['target_word']]),
        #         'synonym': torch.Tensor([sample['synonym']]),
        #         'antonym': torch.Tensor([sample['antonym']]) }
        return (torch.LongTensor([sample[0]]),
                torch.LongTensor([sample[1]]),
                torch.LongTensor([sample[2]]))


class WordnetTriples(Dataset):
    def __init__(self, raw_triples, w2i, i2w, transform=None):
        self.word_triples = raw_triples
        self.w2i, self.i2w = w2i, i2w
        self.transform = transform

    def __len__(self):
        return len(self.word_triples)

    def __getitem__(self, idx):
        # Gets a triple in the form of words. It looks like
        #    ('good', 'goodness', 'badness')
        sample = self.word_triples[idx]

        # Transform the triple into word indexes. Now it looks like
        #    (101, 324, 671)
        # sample = {'target_word': self.w2i[sample[0]],
        #           'synonym': self.w2i[sample[1]],
        #           'antonym': self.w2i[sample[2]] }
        target_word = self.w2i.get(sample[0], 3)
        synonym = self.w2i.get(sample[1], 3)
        antonym = self.w2i.get(sample[2], 3)
        sample = (target_word, synonym, antonym)

        # Maybe transform further the value
        if self.transform:
            sample = self.transform(sample)

        return sample

def get_triples_loader(data_path, dataset_name, w2i, i2w,
                       batch_size, shuffle,
                       num_workers=4,
                       validation_split=0.05):
    train_file = os.path.join(data_path, 'datasets', dataset_name, 'train.pkl')
    test_file = os.path.join(data_path, 'datasets', dataset_name, 'test.pkl')

    train_and_val_triples = pickle.load(open(train_file, 'rb'))

    train_triples, val_triples = split_triples(train_and_val_triples,
                                               validation_split)

    train_dataset = WordnetTriples(train_triples, w2i, i2w, ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    val_dataset = WordnetTriples(val_triples, w2i, i2w, ToTensor())
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)

    test_triples = pickle.load(open(test_file, 'rb'))
    test_dataset = WordnetTriples(test_triples, w2i, i2w, ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch language model')
    parser.add_argument('dataset_path', type=str,
                        help='')
    return parser.parse_args()


def main(args):
    vocab_path = os.path.join(args.dataset_path, 'vocab.txt')
    w2i, i2w = load_vocabulary(vocab_path)

    train_dl, val_dl, test_dl = get_triples_loader(args.dataset_path, w2i, i2w,
                                            batch_size=128, shuffle=True)

    for i_batch, sample_batched in enumerate(train_dl):
        print(i_batch, sample_batched[0].size(),
              sample_batched[1].size(), sample_batched[2].size())

    # for i in range(10):
    #     s = dataset[i]
    #     print(i, 'tw: ', s[0], 'syn: ', s[1], 'ant: ', s[2])


if __name__ == '__main__':
    # Test the functionalities of the dataset
    args = parse_args()
    main(args)
