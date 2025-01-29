import torch
import torch.nn as nn
from s2s_model import Seq2SeqAttentionModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_dataloader, 
          inp_vocabsize,
          tar_vocabsize, 
          embedding_size=128,
          hidden_size=128,
          epoch=10,
          max_len=7):

    model = Seq2SeqAttentionModel(embedding_size,
                                hidden_size,
                                inp_vocabsize,
                                tar_vocabsize,
                                max_len=max_len)
    model.to(device)
    criteria = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.RMSprop(model.parameters())


    for epoch_num in range(epoch):
        total_loss = 0
        for input_sentance, target_sentance in train_dataloader:
            input_sentance,  target_sentance = input_sentance.to(device), target_sentance.to(device)
            output = model(input_sentance,target_sentance)
            output.to(device)
            target_sentance = target_sentance.view(-1)
            output = output.view(target_sentance.shape[0],tar_vocabsize)
            optimizer.zero_grad()
            loss = criteria(output,target_sentance)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'loss in {epoch_num} epoch:',total_loss)
    
    return model

