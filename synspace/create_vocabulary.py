# Given the pickle file containing all the triples, creates a new vocabulary

import os
import argparse
import pickle

import synspace.vocabulary_utils

def pickle2txt(input_file, output_file):
    p = pickle.load(open(input_file, "rb"))
    with open(output_file, 'w') as out:
        for i in p:
            out.write('{}\t{}\t{}\n'.format(*i))

def parse_args():
    parser = argparse.ArgumentParser(description='Creates a vocabulary')
    parser.add_argument('input_file', type=str,
                        help='Pickle file containing all the tuples')

    args = parser.parse_args()
    return args

def main(args):
    # Creates a file that the vocabulary functions can handle
    in_txt = args.input_file + '.txt'
    pickle2txt(args.input_file, in_txt)

    # Call vocabulary functions
    dir_name = os.path.dirname(in_txt)
    synspace.vocabulary_utils.new_vocabulary([in_txt], dir_name, 1, 'spacy',
                False, 1000000, 'word_triples',
                lambda line: " ".join(line.split('\t')))

if __name__ == '__main__':
    args = parse_args()
    main(args)
