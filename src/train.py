import time
import json

import torch
import pandas as pd
import numpy as np

import config, model, dataset, engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

from torch.optim import AdamW

def train_run():
    df = pd.read_csv(config.DATA_PATH)
    
    # 10% of 53,951 for test set
    df_train, df_valid = train_test_split(df, test_size=0.1, random_state=7)
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    # train dataset dataloader
    train_dataset = dataset.BERTDataset(
        text=df_train.text.values, 
        targets=df_train.drop(columns={'text'}).values,
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    
    # test dataset dataloader
    valid_dataset = dataset.BERTDataset(
        text=df_valid.text.values, 
        targets=df_valid.drop(columns={'text'}).values,
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE
    )
    
    device = torch.device(config.DEVICE)
    bert_model = model.BERTMultiLabel()
    bert_model.to(device)
    
    optimizer = AdamW(list(bert_model.parameters()), lr= config.LEARNING_RATE)
    all_metrics = {}
    for epoch in range(config.EPOCHS):
        temp = {}
        # training
        engine.train_fn(train_data_loader, bert_model, optimizer, device)
        # evaluation
        outputs, targets = engine.eval_fn(valid_data_loader, bert_model, device)
        # threshold fixed as 0.5 while training
        outputs = (np.array(outputs) >= 0.5).astype(int)
        
        # metrics
        temp['precision_macro_avg'] = precision_score(targets, outputs, average='macro')
        temp['recall_macro_avg'] = recall_score(targets, outputs, average='macro')
        temp['f1_macro_avg'] = f1_score(targets, outputs, average='macro')
        temp['hamming_loss'] = hamming_loss(targets, outputs)

        all_metrics[f'epoch-{epoch+1}'] = temp
        torch.save(bert_model.state_dict(), config.MODEL_PATH + f'/bert-weights-epoch-{epoch+1}.pt')
    # saving the model
    torch.save(bert_model.state_dict(), config.MODEL_PATH + '/bert-weights.pt')
    # saving the metrics dict
    with open(config.METRICS_PATH, 'w') as file:
        json.dump(all_metrics, file)

if __name__ == "__main__":
    start = time.time()
    train_run()
    end = time.time()
    time_elapsed = end - start
    print(f'training time: ~ {time_elapsed // 60} mins')