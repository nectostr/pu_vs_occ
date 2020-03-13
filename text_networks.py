from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torchtext.datasets.text_classification import TextClassificationDataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import random_split

import pickle
from tqdm import tqdm
import logging
import time
import numpy as np
import pandas as pd

NGRAMS = 2
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pd_iterator(df_torch, ngrams, yield_cls=False):
    tokenizer = get_tokenizer(None)
    for row_id in range(len(df_torch)):
        tokens = df_torch.iloc[row_id]["text"]
        tokens = tokenizer(tokens)
        if yield_cls:
            yield df_torch.iloc[row_id]["label"], ngrams_iterator(tokens, ngrams)
        else:
            yield ngrams_iterator(tokens, ngrams)

def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                        for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


def _setup_datasets(df_torch, ngrams=1, vocab=None, include_unk=False):
    if vocab is None:
        vocab = build_vocab_from_iterator(_pd_iterator(df_torch, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")

    train_data, train_labels = _create_data_from_iterator(
        vocab, _pd_iterator(df_torch, ngrams, yield_cls=True), include_unk)

    return TextClassificationDataset(vocab, train_data, train_labels)

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, 16)
#         self.fc2 = nn.Linear(embed_dim//2, embed_dim//2)
        self.fc3 = nn.Linear(16, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
#         self.fc2.weight.data.uniform_(-initrange, initrange)
#         self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc1(embedded)
        x = torch.sigmoid(x)
#         x = self.fc2(x)
#         x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    try:
        text = torch.cat(text)
    except RuntimeError:
        print(text)
        raise
    return text, offsets, label



def get_text_result(df):
    train_dataset = _setup_datasets(df, ngrams=NGRAMS, include_unk=True)
    # with open("../train_ds_lemm.pckl", 'rb') as f:
    #     train_dataset = pickle.load(f)
    # with open("../test_ds_lemm.pckl", 'rb') as f:
    #     test_dataset = pickle.load(f)
    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = 32
    NUM_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, 1).to(device)

    from torch.utils.data import DataLoader

    def train_func(sub_train_):

        # Train the model
        train_loss = 0
        train_acc = 0
        data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=generate_batch)
        for i, (text, offsets, cls) in enumerate(data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device).squeeze()
            output = model(text, offsets).squeeze()
            loss = criterion(output, cls.float())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.round().int() == cls).sum().item()

        # Adjust the learning rate
        #     scheduler.step()

        return train_loss / len(sub_train_), train_acc / len(sub_train_)

    def test(data_):
        loss = 0
        acc = 0
        data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
        for text, offsets, cls in data:
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device).squeeze()
            with torch.no_grad():
                output = model(text, offsets).squeeze()
                loss = criterion(output, cls.float())
                loss += loss.item()
                acc += (output.round().int() == cls).sum().item()

        return loss / len(data_), acc / len(data_)

    def predict(data_):
        result = []
        data = DataLoader(data_, batch_size=1, collate_fn=generate_batch)

        for text, offsets, _ in data:
            text, offsets = text.to(device), offsets.to(device)
            with torch.no_grad():
                output = model(text, offsets)
                result.append(output.cpu().numpy()[0][0])

        return np.array(result)

    import time
    from torch.utils.data.dataset import random_split
    N_EPOCHS = 20
    min_valid_loss = float('inf')

    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test(sub_valid_)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        # print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        # print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        # print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    return predict(train_dataset)


if __name__ == '__main__':
    df = pd.read_csv("DATA/text_test/malicious_posts.csv", header=None)
    df.columns = "index", "text", "label1", "label2"
    df["text"] = (df["text"]
                  .str
                  .lower()
                  .replace(to_replace='[^A-Za-zА-Яа-я ]+', value=' ', regex=True)
                  .replace(to_replace=' +', value=' ', regex=True))
    df.drop(df[df['text'].map(len) < 10].index, inplace=True)
    label_map = {j: float(i) for i, j in enumerate(set(df["label1"]))}
    df_torch = pd.DataFrame(df["label1"].apply(lambda x: label_map[x]))
    df_torch["text"] = df["text"]
    get_text_result(df_torch)