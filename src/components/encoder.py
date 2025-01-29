import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
  def __init__(self, 
               embedding_size, 
               hidden_size, 
               vocab_size):
    super().__init__()

    # encoder initialization
    self.embedding = nn.Embedding(num_embeddings=vocab_size,
                             embedding_dim=embedding_size)
    self.encoder_lstm = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           batch_first=True)

  def forward(self,input):
    # network flow
    embedding_input = self.embedding(input)
    encoder_outputs, (final_hidden_state, final_cell_state) = self.encoder_lstm(embedding_input)
    
    return encoder_outputs.to(device), final_hidden_state.to(device), final_cell_state.to(device)


