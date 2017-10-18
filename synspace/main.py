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


def euclidean_similarity(w1, w2):
    return ((w1 - w2) ** 2).sum(dim=1)

def get_similarity(similarity_str):
    return {
        'euclidean': euclidean_similarity
    }.get(similarity_str)

def loss_function(x, similarity, args):
    target_word, synonym, antonym = x

    # Calculate the similarity between all elements
    good_similarity = similarity(target_word, synonym)
    bad_similarity  = similarity(target_word, antonym)

    loss = F.relu(args.margin - good_similarity + bad_similarity)
    return loss.sum()


def prepare_for_model(batch, args):
    for i in range(3):
        batch[i] = Variable(batch[i])
        if args.cuda:
            batch[i] = batch[i].cuda()

    return batch


def train(model, optimizer, args, train_loader, val_loader, experiment_address):
    nb_epoch = args.nb_epoch
    similarity = get_similarity(args.similarity)

    val_loss = []
    best_loss = float('inf')

    curr_iter = 0
    for epoch in range(1, nb_epoch+1):
        print('Fitting epoch %d' % epoch, file=sys.stderr)

        train_loss = 0
        for batch_idx, sample_batched in enumerate(train_loader):
            data = prepare_for_model(sample_batched, args)
            x = model(data)

            loss = loss_function(x, similarity, args)
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

                    loss = loss_function(x, similarity, args)
                    curr_val_loss += loss.data[0]

                    print('Validation: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        epoch, val_batch_idx * args.batch_size,
                        len(val_loader.dataset),
                        100. * val_batch_idx / len(val_loader),
                        loss.data[0] / len(data)))

                print('Final Validation Loss: {}'.format(loss.data[0]))

                val_loss.append(curr_val_loss)

                is_best = True if val_loss[-1] < best_loss else False
                if is_best:
                    # If I found a good model, then I totally want to dump it
                    best_loss = val_loss[-1]
                    dump_model({'model': model.state_dict(),
                                'iter': curr_iter},
                               experiment_address,
                               'chkpnt_{}.pth.tar'.format(curr_iter),
                               is_best=is_best)

            if curr_iter % args.save_every == 0:
                dump_model({'model': model.state_dict(),
                            'iter': curr_iter},
                           experiment_address,
                           'chkpnt_{}.pth.tar'.format(curr_iter),
                           is_best=False)

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


def dump_model(info, file_path, file_name, is_best):
    # Call this functions with something like:
    #
    # dump_model({
    #     'iter' : curr_iter,
    #     'model': model.state_dict(),
    # }, 'some_file_name.tar', False)
    #
    file = os.path.join(file_path, file_name)
    torch.save(info, file)
    if (is_best):
        best_file = os.path.join(file_path, 'best.pth.tar')
        torch.save(info, best_file)
        best_list_file = os.path.join(file_path, 'best_list.txt')
        with open(best_list_file, 'a') as f:
            f.write('{}\t{}\n'.format(file_name, info['iter']))


def load_model(filename, model):
    if not os.path.isfile(filename):
        print("=> no checkpoint found at {}".format(filename))
        return None
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    return model


def initialize_vocabulary(data_path, dataset_name):
    dataset_path = os.path.join(data_path, 'datasets', dataset_name)
    vocab_file = os.path.join(dataset_path, 'vocab.txt')
    w2i, i2w = load_vocabulary(vocab_file)
    w2v = preload_w2v(w2i)
    return w2i, i2w, w2v


def initialize_model(w2i, i2w, w2v):
    # "Builds" the model and sends it to the GPU
    model = LanguageModel(w2i, i2w, w2v)
    if args.cuda:
        model.cuda()
    return model


def initialize_or_load_experiment(args, w2i, i2w, w2v):
    experiments_path = os.path.join(args.data_path, 'experiments',
                                   args.dataset_name)
    experiment_name = '{}_{}_{}neu_{}drop_{}reg_{}'.format(
                            args.layers, args.layer_type, args.layer_size,
                            args.dropout, args.l2_reg, args.similarity)
    experiment_address = os.path.join(experiments_path, experiment_name)
    experiment_best_file = os.path.join(experiment_address, 'best.pth.tar')

    if os.path.exists(experiment_best_file):
        if args.mode == 'train':
            print("=> ERROR: No support for resuming training.")
            print("      Experiment exists. Aborting...")
            sys.exit()
        model = LanguageModel(w2i, i2w, w2v)
        model = load_model(experiment_best_file, model)
        model.eval()
    else:
        if not os.path.exists(experiment_address):
            os.makedirs(experiment_address)
        model = initialize_model(w2i, i2w, w2v)
    return model, experiment_address


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch language model')
    parser.add_argument('dataset_name', type=str,
                        help='Name of the dataset to be processed.')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Path to experiment data.')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode of execution: `train` or `test`.')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='How close two words need to be to be fully '
                             'taken as synonyms.')

    parser.add_argument('--layers', type=int, default=2,
                        help='How many layers will the model have?')
    parser.add_argument('--layer_type', type=str, default='relu',
                        help='Either "sigm" or "relu".')
    parser.add_argument('--layer_size', type=int, default=64,
                        help='How many neurons in each layer?')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='How much dropout in each layer?')
    parser.add_argument('--l2_reg', type=float, default=0,
                        help='How much L2 regulation?')
    parser.add_argument('--similarity', type=str, default='euclidean',
                        help='Similarity metric to be used for the margin loss')

    # parser.add_argument('--layers', type=int,
    #                     help='How many layers will the model have?')
    # parser.add_argument('--layer_type', type=str,
    #                     help='Either "sigm" or "relu".')
    # parser.add_argument('--layer_size', type=int,
    #                     help='How many neurons in each layer?')
    # parser.add_argument('--dropout', type=float,
    #                     help='How much dropout in each layer?')
    # parser.add_argument('--l2_reg', type=float,
    #                     help='How much L2 regulation?')
    # parser.add_argument('--similarity', type=str,
    #                     help='Similarity metric to be used for the margin loss')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--nb_epoch', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--language', type=str, default='en',
                        help='Language code to be used. Default: "en";'
                             ' Accepted: "en", "de".')
    parser.add_argument('--validation_split', type=float, default=0.05,
                        help='how much of the training set should be used for '
                             'validation')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='shuffle the training set?')

    parser.add_argument('--log_every', type=int, default=1,
                        help='After how many iterations should we print state?')
    parser.add_argument('--validate_every', type=int, default=37,
                        help='After how many iterations should we run validation?')
    parser.add_argument('--save_every', type=int, default=50,
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

    w2i, i2w, w2v = initialize_vocabulary(args.data_path, args.dataset_name)

    train_loader, val_loader, test_loader = get_triples_loader(
                                args.data_path, args.dataset_name, w2i, i2w,
                                args.batch_size, args.shuffle,
                                validation_split=args.validation_split)

    model, experiment_address = initialize_or_load_experiment(args, w2i, i2w, w2v)

    # Defines a new optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=1e-3)

    if args.mode == 'train':
        train(model, optimizer, args,
              train_loader, val_loader, experiment_address)
    elif args.mode == 'test':
        #test(model, optimizer, args, test_loader)
        pass


if __name__ == '__main__':
    args = parse_args()
    main(args)

