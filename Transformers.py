#!/usr/bin/env python
# coding: utf-8
from torch.utils.data import DataLoader
import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from dataset import Triage
from models import BERTClass

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05


def load_data(input_path='data/train_clean.csv'):
    df = pd.read_csv(input_path)
    df = df[['clean_txt', 'Intencion']]
    df.columns = ['Pregunta', 'Intencion']

    encode_dict = {}

    def encode_cat(x):
        if x not in encode_dict.keys():
            encode_dict[x] = len(encode_dict)
        return encode_dict[x]

    df['ENCODE_CAT'] = df['Intencion'].apply(lambda x: encode_cat(x))
    NB_CLASS = len(encode_dict)
    return df, encode_dict, NB_CLASS


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    return tokenizer


def get_loaders(df, tokenizer, validation):
    training_loader, testing_loader = None, None
    if validation:
        train_index, test_index = train_test_split(list(df.index), random_state=13571113)
        # Creating the dataset and dataloader for the neural network
        train_dataset = df.iloc[train_index].reset_index().drop(columns="index")
        test_dataset = df.iloc[test_index].reset_index().drop(columns="index")
        print("FULL Dataset: {}".format(df.shape))
        print("TRAIN Dataset: {}".format(train_dataset.shape))
        print("TEST Dataset: {}".format(test_dataset.shape))
    else:
        train_dataset = df.reset_index().drop(columns="index")
        print("FULL Dataset: {}".format(df.shape))
        print("TRAIN Dataset: {}".format(train_dataset.shape))

    training_set = Triage(train_dataset, tokenizer, MAX_LEN)
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0}
    training_loader = DataLoader(training_set, **train_params)
    if validation:
        testing_set = Triage(test_dataset, tokenizer, MAX_LEN)
        test_params = {'batch_size': VALID_BATCH_SIZE,
                       'shuffle': True,
                       'num_workers': 0}
        testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader


def evaluate(model, device, data_loader, validation):
    if not validation:
        print("Not validation Data")
        return
    model.eval()
    y_true = []
    y_pred = []
    n_correct = 0
    total = 0
    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            big_val, big_idx = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            n_correct += (big_idx == targets).sum().item()
            y_true += targets
            y_pred += big_idx

        y_true = [y.cpu().numpy().tolist() for y in y_true]
        y_pred = [y.cpu().numpy().tolist() for y in y_pred]
        bacc = balanced_accuracy_score(y_true, y_pred) * 100
        acc = (n_correct*100.0) / total

        print("Accuracy: %0.2f%%" % acc)
        print("Balanced Accuracy: %0.2f%%" % bacc)


def train(df, nb_class, output_model_file, output_vocab_file, validation):
    tokenizer = load_tokenizer()

    training_loader, testing_loader = get_loaders(df, validation)

    device = 'cuda'  # 'cpu'
    model = BERTClass(nb_class)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for _, data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_function(outputs, targets)

            if _%1000 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluate(model, device, training_loader, validation)
        evaluate(model, device, testing_loader, validation)

    torch.save(model, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)
    print('All files saved')


if __name__ == "__main__":
    df, encode_dict, nb_class = load_data(input_path='data/train_clean.csv')
    train(df,
          nb_class,
          validation=True,
          output_model_file='./clean_text/pytorch_beto_news_clean.bin',
          output_vocab_file='./clean_text/vocab_beto_news_clean.bin')
