from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import spacy

from synspace.vocabulary_utils import load_vocabulary, preload_w2v

import os
import sys

from synspace.model import LanguageModel
from synspace.wordnet_triples import get_triples_loader


def euclidean_distance(w1, w2):
    return ((w1 - w2) ** 2).sum(dim=1)

def get_distance(distance_str):
    return {
        'euclidean': euclidean_distance
    }.get(distance_str)

def loss_function(x, distance, args):
    target_word, synonym, antonym = x

    # Calculate the distance between all elements
    good_distance = distance(target_word, synonym)
    bad_distance  = distance(target_word, antonym)

    loss = F.relu(args.margin - good_distance + bad_distance)
    return loss.sum()


def prepare_for_model(batch, args):
    for i in range(3):
        batch[i] = Variable(batch[i])
        if args.cuda:
            batch[i] = batch[i].cuda()

    return batch


def train(model, optimizer, args, train_loader, val_loader):
    nb_epoch = args.nb_epoch
    distance = get_distance(args.distance)

    val_loss = []
    curr_iter = 0
    for epoch in range(1, nb_epoch+1):
        print('Fitting epoch %d' % epoch, file=sys.stderr)

        train_loss = 0
        for batch_idx, sample_batched in enumerate(train_loader):
            data = prepare_for_model(sample_batched, args)
            x = model(data)

            loss = loss_function(x, distance, args)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()

            curr_iter += 1
            if curr_iter % args.log_every == 0:
                 print('Train Iter: {}, Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                     curr_iter,
                     epoch,
                     batch_idx * args.batch_size,
                     len(train_loader.dataset),
                     100. * batch_idx / len(train_loader),
                     loss.data[0] / len(data)))

            if curr_iter % args.validate_every == 0:
                curr_val_loss = 0
                for val_batch_idx, val_sample_batched in enumerate(val_loader):
                    data = prepare_for_model(val_sample_batched, args)
                    x = model(data)

                    loss = loss_function(x, distance, args)
                    curr_val_loss += loss.data[0]

                    print('Validation: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        epoch, val_batch_idx * args.batch_size,
                        len(val_loader.dataset),
                        100. * val_batch_idx / len(val_loader),
                        loss.data[0] / len(data)))

                print('Final Validation Loss: {}'.format(loss.data[0]))

                val_loss.append(curr_val_loss)

    return val_loss

# def test(model, optimizer, nlp, args):
#     model.eval()
#     test_loss = 0
#     for data, _ in test_loader:
#         if args.cuda:
#             data = data.cuda()
#         data = Variable(data, volatile=True)
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

def dump_model(model, filename):
    torch.save(model, filename)

def load_model(filename):
    # if not os.path.isfile(filename):
    #     print("=> no checkpoint found at {}".format(filename))
    #     return
    pass


def initialize_vocabulary(dataset_path):
    vocab_file = os.path.join(dataset_path, 'vocab.txt')
    w2i, i2w = load_vocabulary(vocab_file)
    w2v = preload_w2v(w2i)
    return w2i, i2w, w2v


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch language model')
    parser.add_argument('dataset_path', type=str,
                        help='Path to dataset to be processed.')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode of execution: `train` or `test`.')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='How close two words need to be to be fully '
                             'taken as synonyms.')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--nb_epoch', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--language', type=str, default='en',
                        help='Language code to be used. Default: "en";'
                             ' Accepted: "en", "de".')
    parser.add_argument('--distance', type=str, default='euclidean',
                        help='Distance metric to be used for the margin loss')
    parser.add_argument('--validation_split', type=float, default=0.05,
                        help='how much of the training set should be used for '
                             'validation')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='shuffle the training set?')

    parser.add_argument('--log_every', type=int, default=1,
                        help='After how many iterations should we print state?')
    parser.add_argument('--validate_every', type=int, default=500,
                        help='After how many iterations should we run validation?')
    parser.add_argument('--save_every', type=int, default=1,
                        help='After how many iterations should we save state?')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main(args):
    # Initializes the random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    w2i, i2w, w2v = initialize_vocabulary(args.dataset_path)

    train_loader, val_loader, test_loader = get_triples_loader(
                                args.dataset_path, w2i, i2w,
                                args.batch_size, args.shuffle,
                                validation_split=args.validation_split)

    #for i_batch, sample_batched in enumerate(train_loader):
    #    print(i_batch, sample_batched[0].size(),
    #          sample_batched[1].size(), sample_batched[2].size())

    # "Builds" the model and sends it to the GPU
    model = LanguageModel(w2i, i2w, w2v)
    if args.cuda:
        model.cuda()

    # Defines a new optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=1e-3)

    if args.mode == 'train':
        train(model, optimizer, args, train_loader, val_loader)
    elif args.mode == 'test':
        #test(model, optimizer, args, test_loader)
        pass


if __name__ == '__main__':
    args = parse_args()
    main(args)

