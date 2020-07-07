import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter


def load_data(input_path='data/train_clean.csv', columns_Q='clean_txt', weight=False):
    df = pd.read_csv(input_path)
    df = df[[columns_Q, 'Intencion']]
    df.columns = ['Pregunta', 'Intencion']

    encode_dict = {}

    def encode_cat(x):
        if x not in encode_dict.keys():
            encode_dict[x] = len(encode_dict)
        return encode_dict[x]

    df['ENCODE_CAT'] = df['Intencion'].apply(lambda x: encode_cat(x))
    NB_CLASS = len(encode_dict)
    
    weight_list = None
    if weight:
        class_counter = Counter(df['ENCODE_CAT'])
        weight_list = [1 / class_counter[i] for i in range(NB_CLASS)]

    return df, encode_dict, NB_CLASS, weight_list

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, mode="train"):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __getitem__(self, index):
        pregunta = str(self.data.Pregunta[index])
        pregunta = " ".join(pregunta.split())
        inputs = self.tokenizer.encode_plus(
            pregunta,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        d = {'ids': torch.tensor(ids, dtype=torch.long),
             'mask': torch.tensor(mask, dtype=torch.long),
             'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)}
        col_target = "id" if self.mode == "submit" else "ENCODE_CAT"
        d['targets'] = torch.tensor(self.data[col_target][index], dtype=torch.long)
        return d

    def __len__(self):
        return self.len
