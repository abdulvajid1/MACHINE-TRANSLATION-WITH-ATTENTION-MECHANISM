import torch
import torch.nn as nn

from components.encoder import Encoder
from components.decoder import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Seq2SeqAttentionModel(nn.Module):
  def __init__(self,
               embedding_size,
               hidden_size,
               vocab_size_en,
               vocab_size_tr,
               max_len=10):
    super().__init__()

    self.encoder = Encoder(embedding_size,
                           hidden_size,
                           vocab_size_en).to(device)

    self.decoder = Decoder(embedding_size,
                           hidden_size,
                           vocab_size_tr,
                           max_len,
                           sos_token=1).to(device)

  def forward(self, input, target_output):
    encoder_outputs = self.encoder(input)
    decoder_output = self.decoder(*encoder_outputs, target_output)

    return decoder_output