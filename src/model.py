import transformers
import torch.nn as nn

class BERTMultiLabel(nn.Module):
    def __init__(self):
        super(BERTMultiLabel, self).__init__()
        # pretrained model (all layers freezed)
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        # adding a dropout layer (30%)
        self.drop = nn.Dropout(0.3)
        self.output = nn.Linear(768, 28)
        
    def forward(self, ids, mask, token_type_ids):
        output = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        dropout = self.drop(output.pooler_output)
        output_logits = self.output(dropout)
        return output_logits