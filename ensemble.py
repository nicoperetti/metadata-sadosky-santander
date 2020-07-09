import pandas as pd
import numpy as np
from scipy import stats

ENSEMBLES = {
    0: {"input": ["submits/countv_tfidf_svc_df_0.68_lb.csv",
                  "submits/preprocess_countv_tfidf_svc_df_0.71.csv",
                  "submits/preprocess_countv_tfidf_mlp_df_0.70.csv",
                  "submits/submit_transfomer_0.8145_lb.csv",  # without weights;without DA
                  "submits/submit_transfomer_0.846.csv"],  # with weights; with DA (es,en)
        "output": "submits/submit_ensemble_nb_0.csv",  # lb 0.7829
        "gold": 4},
    1: {"input": ["submits/preprocess_countv_tfidf_svc_df_0.71.csv",
                  "submits/submit_transfomer_0.8145_lb.csv",  # without weights;without DA
                  "submits/submit_transfomer_0.846.csv"],  # with weights; with DA (es,en)
        "output": "submits/submit_ensemble_nb_1.csv",  # lb 0.850
        "gold": 2}
}


def ensemble(preds_f, output_path, gold, option):
    if option == "mode":
        values = []
        for pred_f in preds_f:
            df = pd.read_csv(pred_f, header=None)
            values.append(df[1].values)
        preds = np.array(values)
        gold_preds = preds[gold, :]

        mode, counts = stats.mode(preds)
        final_preds = mode[0]
        counts = counts[0]

        to_gold_idxs = list(np.argwhere(counts == 1).reshape(1, -1)[0])

        final_preds[to_gold_idxs] = gold_preds[to_gold_idxs]

        df_ensemble = pd.DataFrame(list(final_preds))
        df_ensemble.to_csv(output_path, header=False)


if __name__ == "__main__":
    ens = 0
    ensemble(ENSEMBLES[ens]["input"],
             ENSEMBLES[ens]["output"],
             gold=ENSEMBLES[ens]["gold"],
             option="mode")
