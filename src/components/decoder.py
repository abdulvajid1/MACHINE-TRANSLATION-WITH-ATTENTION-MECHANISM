import torch
import torch.nn as nn

from attention import Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Decoder(nn.Module):
  def __init__(self,
               embedding_dim,
               hidden_size,
               vocab_size_tr,
               max_len=20,
               sos_token=1,
               ):
    super().__init__()
    
    self.MAX_LEN = max_len
    self.SOS_TOKEN = sos_token

    # Layers Initialization
    self.embedding_layer = nn.Embedding(vocab_size_tr, embedding_dim)
    self.lstm_layer = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
    self.fnn = nn.Linear(hidden_size ,vocab_size_tr)
    self.attention_vector = Attention(hidden_size)

  def forward(self,
              encoder_outputs,
              hidden_state,
              cell_state,
              target_output=None):
    
    batch_size = encoder_outputs.shape[0]     # encoder gets the input from train loader which defines the batchsize
    decoder_input = torch.empty(size=(batch_size,1),dtype=torch.long).fill_(self.SOS_TOKEN).to(device)     # Initialize first input [32 sos_tokens]
    decoder_outputs = []

    for i in range(self.MAX_LEN):
      output_logits ,hidden_state, cell_state = self.forward_step(encoder_outputs, decoder_input, hidden_state, cell_state)
      decoder_outputs.append(output_logits.unsqueeze(1))      # decoder ouput = [(32,vocab_size),...(32,vocab_size)], this list will have max_len item , lastly we will concat this to make (32,max_len,vocab_size)

      # teacher_forcing, occurs if we give target_output in the decoder
      if target_output != None:
        decoder_input = target_output[:,i].unsqueeze(1)
      else:
        _, decoder_input = output_logits.topk(1,dim=-1)

    decoder_final_output = torch.cat(decoder_outputs,dim=1)
    return decoder_final_output

  def forward_step(self,encoder_outputs, decoder_input, hidden_state, cell_state):
    embedded_decoder_input = self.embedding_layer(decoder_input)      # embedded shape : (32,1,embedd_size), here 1 , becuase we are giving each word or token to decoder and make it predict next word
    lstm_output, (decoder_hidden, decoder_cell) = self.lstm_layer(embedded_decoder_input, (hidden_state, cell_state))     # lstm_output: (32,1,hidden_size)
    output_logit = self.fnn(lstm_output.squeeze(1))   # squeeze (32,1,hidden_size) -> 32,hidden_size
    hidden_state = self.attention_vector(encoder_outputs, decoder_hidden)     # ouput_logits: (32,vocab_size) , 32 prediction of word , we will pic top item

    return output_logit ,hidden_state , decoder_cell