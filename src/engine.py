import torch
import torch.nn as nn
from tqdm import tqdm

# loss function
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

# train function
def train_fn(data_loader, model, optimizer, device):
    # train mode
    model.train()
    
    for d in tqdm(data_loader, total=len(data_loader)):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']
        
        # switch device only as they already are in tensor form
        ids = ids.to(device)
        token_type_ids = token_type_ids.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        
        # forward pass
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        
        # loss calculation
        loss = loss_fn(outputs, targets)
        
        # zero optimizer gradients
        optimizer.zero_grad()
        
        # backpropagation
        loss.backward()
        
        # update gradients
        optimizer.step()


# eval function
def eval_fn(data_loader, model, device):

    # evaluation mode
    model.eval()
    fin_targets = []
    fin_outputs = []
    
    # inference mode
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        # forward pass
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        
    return fin_outputs, fin_targets