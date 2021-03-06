import click
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


@click.command()
@click.option('--input_path', type=click.STRING, help='Path to input file')
@click.option('--output_dir', type=click.STRING, help='Path to output dir')
@click.option('--tdd', type=click.BOOL, help='TDD: Test time Data Augmentation')
def predict_script(input_path, output_dir, tdd):
    mapping_dict = output_dir + "mapping.json"
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(input_path)

    encode_dict = json.load(open(mapping_dict, "r"))
    encode_dict_inv = {v: k for k, v in encode_dict.items()}

    tokenizer = load_tokenizer()
    model = torch.load(output_dir + "pytorch_beto_news.bin")
    columns = ['clean_txt']

    # Test Time Data Augmentation.
    if tdd:
        columns += ['clean_txt_T1', 'clean_txt_T2_fr', 'clean_txt_T3_pt', 'clean_txt_T4_ar']

    preds_list = []
    id_list = []
    gold_idx = 0
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
    gold_preds = preds_list[gold_idx, :]

    mode, counts = stats.mode(preds_list)
    final_preds = mode[0]
    counts = counts[0]
    to_gold_idxs = list(np.argwhere(counts == 1).reshape(1, -1)[0])
    final_preds[to_gold_idxs] = gold_preds[to_gold_idxs]

    df_submit = pd.DataFrame(list(zip(id_list, list(final_preds))))
    df_submit.to_csv(output_dir + "submit_transfomer_BETO.csv",
                     header=False,
                     index=False)


if __name__ == "__main__":
    predict_script()
