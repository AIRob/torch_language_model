import argparse
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from synspace import split_triples
from synspace.vocabulary_utils import load_vocabulary


class ToTensor():
    """Convert tuples in sample to Tensors."""

    def __call__(self, sample):
        return {'target_word': torch.Tensor([sample['target_word']]),
                'synonym': torch.Tensor([sample['synonym']]),
                'antonym': torch.Tensor([sample['antonym']]) }


class WordnetTriples(Dataset):
    def __init__(self, raw_triples, vocab_path, transform=None):
        self.word_triples = raw_triples
        self.w2i, self.i2w = load_vocabulary(vocab_path)

        self.transform = transform

    def __len__(self):
        return len(self.word_triples)

    def __getitem__(self, idx):
        # Gets a triple in the form of words. It looks like
        #    ('good', 'goodness', 'badness')
        sample = self.word_triples[idx]

        # Transform the triple into word indexes. Now it looks like
        #    (101, 324, 671)
        sample = {'target_word': self.w2i[sample[0]],
                  'synonym': self.w2i[sample[1]],
                  'antonym': self.w2i[sample[2]] }

        # Maybe transform further the value
        if self.transform:
            sample = self.transform(sample)

        return sample

def get_triples_loader(input_file, vocab_path,
                       batch_size, shuffle, num_workers):
    raw_triples = pickle.load(open(input_file, "rb"))

    train_triples, val_triples = split_triples(raw_triples, 0.05)

    train_dataset = WordnetTriples(train_triples, vocab_path, ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)

    val_dataset = WordnetTriples(val_triples, vocab_path, ToTensor())
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch language model')
    parser.add_argument('input_file', type=str,
                        help='')
    parser.add_argument('vocab_path', type=str,
                        help='')
    return parser.parse_args()


def main(args):
    dataset = WordnetTriples(args.input_file, args.vocab_path, ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['target_word'].size(),
              sample_batched['synonym'].size(), sample_batched['antonym'].size())

    # for i in range(10):
    #     s = dataset[i]
    #     print(i, 'tw: ', s['target_word'],
    #           'syn: ', s['synonym'], 'ant: ', s['antonym'])


if __name__ == '__main__':
    # Test the functionalities of the dataset
    args = parse_args()
    main(args)
