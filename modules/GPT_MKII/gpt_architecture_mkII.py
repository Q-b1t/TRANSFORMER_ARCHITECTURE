
import torch
import torch.nn as nn
import math
import numpy as np



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


class GTPMKII(nn.Module):
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
    self.positional_encoding = PositionalEncoding(
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
        dropout=dropout_p
    )

    # instance the final layer
    self.out = nn.Linear(embedding_dimention, num_tokens)

  def get_target_mask(self, size):
    """
    Given a size (dimention), the function creates a lower triangular matrix and subsitutes 1 with 0 and 0 with -inf.
    This is to create a masking for sequence tokens during training (for the model not to have access to the entire sequence
    at once but to increment the tokens sequentially).
    """
    mask = torch.tril(torch.ones(size,size) == 1) # genetate the a boolean matrix from a lower triangular matrix
    mask = mask.float() # transform to floating point numbers
    # mask according to the value mapping decribed above
    mask = mask.masked_fill(mask == 0,float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask

  def get_pad_mask(self,matrix,pad_token):
    return (matrix == pad_token)

  def forward(self,source,target,target_mask = None,source_pad_mask = None,target_pad_mask = None):
    # transform the token into a word embedding
    source = self.embedding(source) * math.sqrt(self.embedding_dimention)
    target = self.embedding(target) * math.sqrt(self.embedding_dimention)
    # add the positional encoding
    source = self.positional_encoding(source)
    target = self.positional_encoding(target)
    # permute to obtain size (sequence length, batch size, model dimention)
    source = source.permute(1,0,2)
    target = target.permute(1,0,2)
    # pass everything into the transformer 
    transformer_out = self.transformer(source,target,tgt_mask=target_mask, src_key_padding_mask=source_pad_mask, tgt_key_padding_mask=target_pad_mask)
    out = self.out(transformer_out)
    return out
