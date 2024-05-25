import os
import numpy as np
from sklearn import model_selection
from torch.utils import data
from transformers import AutoTokenizer


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

        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

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
        for sample in self.samples:
            token_vector = tokenizer(sample,
                                     padding='max_length',
                                     max_length=max_length,
                                     truncation=True)
            vectors.append(np.array(token_vector['input_ids']))

        self.vocab_size = tokenizer.vocab_size
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
