import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
  def __init__(self,embedding_dimention,dropout_p,max_len):
    super().__init__()

    # instance a dropout layer
    self.dropout = nn.Dropout(dropout_p)

    # encoding (from formula)
    pos_encoding = torch.zeros(max_len,embedding_dimention)
    pos_list = torch.arange(0,max_len,dtype = torch.float).view(-1,1)
    division = torch.exp(torch.arange(0, embedding_dimention, 2).float() * (-math.log(10000.0)) / embedding_dimention)
    pos_encoding[:,0::2] = torch.sin(pos_list * division)
    pos_encoding[:,1::2] = torch.cos(pos_list * division)

    pos_encoding = pos_encoding.unsqueeze(0).transpose(0,1)
    self.register_buffer("pos_encoding",pos_encoding)

  def forward(self,token_embedding):
    return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class TransformerMKII(nn.Module):
  def __init__(
      self,
      num_tokens, # length of the token
      embedding_dimention, # embedding dimention
      num_heads, # number of concatenated multihead self attention layers
      num_encoder_layers, # num of encoder layers
      num_decoder_layers, # num of decoder layers
      dropout_p, # probability of shutting down random layers at training (regularization)
      pos_encoding_max_len = 5000
  ):
    super().__init__()

    # save important params
    self.embedding_dimention = embedding_dimention

    # instance the positional encoding layer
    self.positional_encoder = PositionalEncoding(
        embedding_dimention = embedding_dimention,
        dropout_p= dropout_p,
        max_len= pos_encoding_max_len
    )

    # instance the initial embedding layer
    self.embedding = nn.Embedding(num_tokens, embedding_dimention)

    # instance the built in transformer
    self.transformer = nn.Transformer(
        d_model= embedding_dimention,
        nhead = num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout_p,
        batch_first = True
    )

    # instance the final layer
    self.out = nn.Linear(in_features = embedding_dimention,out_features= num_tokens)

  def get_target_mask(self, size):

    mask = torch.tril(torch.ones(size,size) == 1) # genetate the a boolean matrix from a lower triangular matrix
    mask = mask.float() # transform to floating point numbers
    # mask according to the value mapping decribed above
    mask = mask.masked_fill(mask == 0,float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask

  def get_pad_mask(self,matrix,pad_token):
    return (matrix == pad_token)

  def generate_sequence(self,context,max_length,device,mask = False):

    self.eval()

    context = context.to(device)
    y_input = torch.tensor([[context[0,-1]]], dtype=torch.long, device=device)

    num_tokens = len(context[0])

    for _ in range(max_length):
        # Get source mask

        if mask:
          target_mask = self.get_target_mask(size=y_input.size(1)).to(device)
          logits = self(context, y_input,target_mask)
        else:
          logits = self(context, y_input)

        if logits.size(1) > 1:
          logits = logits[:,-1,:]
        probs = F.softmax(logits,dim=-1)
        next_index = torch.multinomial(probs[0],num_samples = 1).item()
        next_item = torch.tensor([[next_index]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

    return y_input.view(-1).tolist()

  def forward(self,source,target,target_mask = None,source_pad_mask = None,target_pad_mask = None):
    # transform the token into a word embedding
    source = self.embedding(source) * math.sqrt(self.embedding_dimention)
    target = self.embedding(target) * math.sqrt(self.embedding_dimention)
    source = self.positional_encoder(source)
    target = self.positional_encoder(target)

    transformer_out = self.transformer(source, target, tgt_mask=target_mask, src_key_padding_mask=source_pad_mask, tgt_key_padding_mask=target_pad_mask)
    out = self.out(transformer_out)
    return out
