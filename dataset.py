import re
import os
import numpy as np
from sklearn import model_selection
from nltk import wordpunct_tokenize
from torch.utils import data


def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    tokens = tokens[:-1] # remove last token because it is the year which maybe is not useful
    return tokens


def create_vocab(texts):
    vocab = set()
    for sentence in texts:
        tokens = tokenize(sentence)
        vocab.update(tokens)
    vocab = list(vocab)
    pad_token = '<PAD>'
    unk_token = '<UNK>'
    vocab.append(pad_token)
    vocab.append(unk_token)
    return vocab


class UCISentimentDataset(data.Dataset):

    def __init__(self, root_dir, mode='train', max_length=100):
        yelp_path = os.path.join(root_dir, 'yelp_labelled.txt')
        amazon_path = os.path.join(root_dir, 'amazon_cells_labelled.txt')
        imdb_path = os.path.join(root_dir, 'imdb_labelled.txt')

        sentences = []
        labels = []
        for path in [yelp_path, amazon_path, imdb_path]:
            with open(path, 'r') as f:
                for line in f:
                    sentence, label = line.strip().split('\t')
                    labels.append(int(label))
                    sentences.append(sentence)

        vocab = create_vocab(sentences)
        self.vocab_size = len(vocab)
        pad_token = '<PAD>'
        unk_token = '<UNK>'
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}

        train_samples, val_samples, train_labels, val_labels = model_selection.train_test_split(
            sentences,
            labels,
            test_size=0.2,
            random_state=42
        )

        if mode == 'train':
            self.samples = train_samples
            self.labels = train_labels
        else:
            self.samples = val_samples
            self.labels = val_labels

        vectors = []
        for tokens in self.samples:
            tokens = tokenize(tokens)
            if len(tokens) < max_length:
                num_pad = max_length - len(tokens)
                tokens.extend([pad_token] * num_pad)
            else:
                tokens = tokens[:max_length]
            token_vector = []

            for word in tokens:
                if word in vocab:
                    token_vector.append(self.token2idx[word])
                else:
                    token_vector.append(self.token2idx[unk_token])

            vectors.append(np.array(token_vector))

        self.samples = vectors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        sample = sample.squeeze()

        label = self.labels[item]
        return sample, label


def uci_sentiment_dataloader(root_dir, mode='train', batch_size=32, max_seq_length=100):
    dataset = UCISentimentDataset(root_dir, mode, max_length=max_seq_length)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=mode == 'train'), dataset.vocab_size
