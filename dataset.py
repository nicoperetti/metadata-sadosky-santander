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


def gather_translations(input_path="data/train_with_translations_clean.csv",
                        output_path="data/train_with_translations_clean_all_es_en_fr.csv",
                        option=None):
    columns_t = ["clean_txt_T1", "clean_txt_T2_fr", "clean_txt_T3_pt", "clean_txt_T4_ar"]

    if option == "weight":
        df = pd.read_csv(input_path)
        df = df[columns_t + ["clean_txt", "Intencion"]]
        df0 = df[["clean_txt", "Intencion"]].copy()
        for col in columns_t:
            thr = df0["Intencion"].value_counts().values[0] // 2
            cats_to_popu = [ k for k, v in dict(df0["Intencion"].value_counts()).items() if v < thr]
            print(f'Amount categories to populate : {len(cats_to_popu)}')
            df1 = df[df["Intencion"].isin(cats_to_popu)][[col, "Intencion"]].copy()
            df1.columns = ["clean_txt", "Intencion"]
            df0 = pd.concat([df0, df1])
            df0.to_csv(output_path, index=False)
    else:
        df = pd.read_csv(input_path)
        columns = ["clean_txt"] + columns_t
        df_list = []
        for col in columns:
            asd = df[[col, "Intencion"]]
            asd.columns = ["clean_txt", "Intencion"]
            df_list.append(asd)
        train = pd.concat(df_list)
        train.to_csv(output_path, index=False)


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
