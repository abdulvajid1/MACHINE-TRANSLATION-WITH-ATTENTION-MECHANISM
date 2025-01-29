import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader):
    model.to(device)
    criteria = nn.CrossEntropyLoss(ignore_index=0)
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for input_sentance, target_sentance in dataloader:
            input_sentance,  target_sentance = input_sentance.to(device), target_sentance.to(device)
            output = model(input_sentance,target_sentance)
            target_sentance = target_sentance.view(-1)
            output = output.view(target_sentance.shape[0],-1)
            loss = criteria(output,target_sentance)
            val_loss += loss.item()

    print(f'Evaluation loss:',val_loss)



