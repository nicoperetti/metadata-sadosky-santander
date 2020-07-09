import json
import pandas as pd
import numpy as np
from scipy import stats
import torch
from dataset import Triage
from torch.utils.data import DataLoader
from models import load_tokenizer


MAX_LEN = 512
VALID_BATCH_SIZE = 16


def predict(model, data_loader, device):
    model.eval()
    id_list = []
    preds = []
    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)

            big_val, big_idx = torch.max(outputs.data, dim=1)
            id_list += targets
            preds += big_idx
        id_list = [id_.cpu().numpy().tolist() for id_ in id_list]
        preds = [pred.cpu().numpy().tolist() for pred in preds]
    return id_list, preds


if __name__ == "__main__":
    output_path = "./translations_en_fr_w/"
    input_path = 'data/test_with_translations_clean.csv'
    mapping_dict = output_path + "mapping.json"
    device = "cuda"

    df = pd.read_csv(input_path)

    encode_dict = json.load(open(mapping_dict, "r"))
    encode_dict_inv = {v: k for k, v in encode_dict.items()}

    tokenizer = load_tokenizer()
    model = torch.load(output_path + "pytorch_beto_news.bin")

    # Test Time Data Augmentation.
    preds_list = []
    id_list = []
    columns = ['clean_txt', 'clean_txt_T1', 'clean_txt_T2_fr', 'clean_txt_T3_pt', 'clean_txt_T4_ar']
    for col in columns:
        df0 = df[['id', col]].copy()
        df0.columns = ["id", "Pregunta"]

        test_set = Triage(df0, tokenizer, MAX_LEN, mode="submit")
        test_params = {'batch_size': VALID_BATCH_SIZE,
                       'shuffle': False,
                       'num_workers': 0}
        test_loader = DataLoader(test_set, **test_params)

        id_list, preds = predict(model, test_loader, device)
        preds = [int(encode_dict_inv[pred].split("_")[1]) for pred in preds]
        preds_list.append(preds)
    preds_list = np.array(preds_list)
    mode, counts = stats.mode(preds_list)
    preds_list = list(mode[0])

    df_submit = pd.DataFrame(list(zip(id_list, preds_list)))
    df_submit.to_csv(output_path + "submit_transfomer.csv", header=False, index=False)
