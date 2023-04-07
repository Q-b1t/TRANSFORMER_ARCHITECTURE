import torch
import torch.nn as nn

torch.manual_seed(4444)

class Head(nn.Module):
  def __init__(self,head_size,block_size,embedding_dimention,dropout):
    super().__init__()
    # instance hyperparameters 
    self.head_size = head_size
    self.block_size = block_size
    self.embedding_dimention = embedding_dimention
    self.dropout = dropout

    # instance layers for single self attention head
    self.key = nn.Linear(self.embedding_dimention, self.head_size, bias=False)
    self.query = nn.Linear(self.embedding_dimention, self.head_size, bias=False)
    self.value = nn.Linear(self.embedding_dimention, self.head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
    self.dropout = nn.Dropout(self.dropout)

  def forward(self,x):
    batch,timesteps,channels = x.shape
    k = self.key(x)
    q = self.query(x)

    # compute the attention scores
    wei = q @ k.transpose(-2,-1) * channels ** -0.5 # dot product normalization and normalize to prevent explosion
    wei = wei.masked_fill(self.tril[:timesteps,:timesteps] == 0,float("-inf")) # only include the previous tokes to average
    wei = F.softmax(wei,dim = -1 ) # normalize to 1

    # add the value
    v = self.value(x)
    out = wei @ v

    return out


class BigramLanguageModelMKII(nn.Module):
  def __init__(self,vocab_size,embedding_dimention,block_size,head_size,head_dropout,device):
    super().__init__()
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.embedding_dimention = embedding_dimention
    self.head_size = head_size
    self.head_dropout = head_dropout

    # embedding matrix for each of the tokens
    self.token_embedding = nn.Embedding(self.vocab_size,self.embedding_dimention)
    self.position_embedding = nn.Embedding(self.block_size,self.embedding_dimention)
    self.sa_head = Head(self.embedding_dimention,self.block_size,self.embedding_dimention,self.head_dropout)
    self.lm_head = nn.Linear(in_features=self.embedding_dimention,out_features=self.vocab_size)

  def forward(self,context,targets = None):
    batch,timesteps = context.shape
    # get the logits in shape (BATCH,TIMESTEPS,CHANNELS)
    token_embedding = self.token_embedding(context)
    pos_embedding = self.position_embedding(torch.arange(timesteps,device = device))
    x = token_embedding + pos_embedding
    x = self.sa_head(x)
    logits = self.lm_head(x)
    if targets is None:
      loss = None
    else:
      batch,timesteps,channels = logits.shape
      logits = logits.view(batch*timesteps,channels)
      targets = targets.view(batch*timesteps)
      loss = F.cross_entropy(logits,targets)
    return logits,loss

  def generate(self,context,max_new_tokens):
    for _ in range(max_new_tokens):
      # cut down block size
      context_condition = context[:,-self.block_size:]
      logits,loss = self(context_condition)
      # focus only on the last timestep
      logits = logits[:,-1,:]
      # convert logits intro pobability distribution
      probs = F.softmax(logits,dim=-1)
      # sample from the distribution
      indx_next = torch.multinomial(probs,num_samples = 1)
      context = torch.cat([context,indx_next],dim = 1)
    return context