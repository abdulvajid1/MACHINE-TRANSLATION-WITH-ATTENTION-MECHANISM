import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()

    self.network = nn.Sequential(
          nn.Linear(2*hidden_size,hidden_size),
          nn.SELU(),
          nn.Linear(hidden_size,1),
          nn.Softmax(dim=1)
        )
    
  def forward(self,encoder_outputs,hidden_state):
    """ Concat encoeder_output and hidden_state, encoder_output shape = (32,timestept,hidden_size), hidden_state shape = 32,1,hidden_side
    first we need to make it same shape to concat hidden_state should be 32,timestep hidden_size, timestpe will be repeatation of same one vector from hidden size"""

    encoder_timestep_len = encoder_outputs.size(1)  # hidden_size will be (1,32,hidden_size) according to doc we need to change
    hidden_state = hidden_state.permute(1,0,2) # shape: (32,1,5)
    hidden_repeated = hidden_state.repeat(1,encoder_timestep_len,1) # hidden_state repetation 

    # concat with encoder_output and hidden output
    encoder_hidden_concat = torch.concat((encoder_outputs,hidden_repeated),dim=-1) # shape : 32,timestep,hidden_size*2
    weights = self.network(encoder_hidden_concat) # (32,timestepe,1)
    weights = weights.permute(0,2,1) # for bmm (32,1,timesteps)
    context_vectores = weights.bmm(encoder_outputs) # 32,1,hidden_size
    context_vectores = context_vectores.permute(1,0,2) # convert back to the way lstm take hidden state

    return context_vectores.to(device)