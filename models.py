import torch
from transformers import AutoModelWithLMHead


class BERTClass(torch.nn.Module):
    def __init__(self, nb_class):
        super(BERTClass, self).__init__()
        self.l1 = AutoModelWithLMHead.from_pretrained("dccuchile/bert-base-spanish-wwm-cased").base_model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, nb_class)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
