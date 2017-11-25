import argparse
from nltk.tokenize import TweetTokenizer
import io
from generic.utils.file_handlers import pickle_dump

from guesswhat.data_provider.guesswhat_dataset import Dataset


# wget http://nlp.stanford.edu/data/glove.42B.300d.zip

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating GLOVE dictionary.. Please first download http://nlp.stanford.edu/data/glove.42B.300d.zip')

    parser.add_argument("-data_dir", type=str, default="." , help="Path to VQA dataset")
    parser.add_argument("-glove_in", type=str, default="glove.42B.300d.zip", help="Name of the stanford glove file")
    parser.add_argument("-glove_out", type=str, default="glove_dict.pkl", help="Name of the output glove file")

    args = parser.parse_args()

    print("Loading dataset...")
    set_names = ["train", "test", "valid"]
    datasets = [Dataset(args.data_dir, which_set=set_name) for set_name in set_names]

    tokenizer = TweetTokenizer(preserve_case=False)

    print("Loading glove...")
    with io.open(args.glove_in, 'r', encoding="utf-8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]


    print("Mapping glove...")
    glove_dict = {}
    not_in_dict = {}
    for _set in datasets:
        for g in _set.games:
            words = tokenizer.tokenize(" ".join(g.questions))
            for w in words:
                w = w.lower()
                w = w.replace("'s", "")
                if w in vectors:
                    glove_dict[w] = vectors[w]
                else:
                    not_in_dict[w] = 1

    # Add Yes/No/N-A token
    glove_dict["<yes>"] = vectors["yes"]
    glove_dict["<no>"] = vectors["no"]
    glove_dict["<n/a>"] = vectors["inappropriate"] # Best I could find :)

    print("Number of glove: {}".format(len(glove_dict)))
    print("Number of words with no glove: {}".format(len(not_in_dict)))

    for k in not_in_dict.keys():
        print(k)

    print("Dumping file...")
    pickle_dump(glove_dict, args.glove_out)

    print("Done!")



