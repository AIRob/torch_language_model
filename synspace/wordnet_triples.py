import argparse
import pickle

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
        return (torch.Tensor([sample[0]]),
                torch.Tensor([sample[1]]),
                torch.Tensor([sample[2]]))


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
        # sample = {'target_word': self.w2i[sample[0]],
        #           'synonym': self.w2i[sample[1]],
        #           'antonym': self.w2i[sample[2]] }
        sample = (self.w2i[sample[0]], self.w2i[sample[1]], self.w2i[sample[2]])

        # Maybe transform further the value
        if self.transform:
            sample = self.transform(sample)

        return sample

def get_triples_loader(input_file, vocab_path,
                       batch_size, shuffle,
                       num_workers=4,
                       validation_split=0.05):
    raw_triples = pickle.load(open(input_file, "rb"))

    train_triples, val_triples = split_triples(raw_triples, validation_split)

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
    train_dl, val_dl = get_triples_loader(args.input_file, args.vocab_path,
                                            batch_size=4, shuffle=True)

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
