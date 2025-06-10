import torch
import config

class BERTDataset:
    def __init__(self, text, targets):
        self.text = text
        self.targets = targets
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        
    def __len__(self):
        return len(self.text) # no of samples
    
    def __getitem__(self, item):
        text = str(self.text[item])
        text = ' '.join(text.split())
        
        # bert-tokenize
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        # return in form of tensors
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }
        