

def split_triples(triples, split=0.3):
    # Casting to integer a value+0.5 will round to the closest integer
    split_index = int(len(triples) * split + 0.5)
    train, test = triples[:-split_index], triples[-split_index:]
    return train, test

