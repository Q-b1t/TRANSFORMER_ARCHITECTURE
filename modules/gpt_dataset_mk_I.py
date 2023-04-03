import torch
import torch.nn as nn
from torch.utils.data import Dataset

torch.manual_seed(4444)

# create pytorch custom dataset for the problem at hand
torch.manual_seed(4444)

class GptDatasetMKI(Dataset):
  """
  **THIS CLASS DOES NOT SUPPORT TOKEN PADDING***
  Inputs:
  target_dir: the directory containing the text corpus
  encoding: the encoding that will be used to create the dataset
  sequence_size: the length the the sequence the model will be fed at each batch iteration
  tokenization_mode: can be either "uniform" or "shifted"
    -> uniform: x will be a tensor of dimentions [block_size] and y will be a tensor of dimentions [1]
       where y contains the index of the token that goes after the last token of x
    -> shifted: both x and y will be tensors of dimentions [block_size], but y is shifted one position
       to the right. This means the "i" element of y is the index of the token that goes after the end
       of sequence x[:i]
  """
  def __init__(self,target_dir,encoding = "utf-8",block_size = 8,tokenization_mode = "uniform"):
    self.text_path = target_dir # directory containing the text corpus
    self.encoding = encoding # encoding used to read the text
    self.block_size = block_size # length the sequence to tokenize and parse the samples
    self.token_mode = tokenization_mode # can be "uniform" or "shifted"
    # retrieve the text corpus from the target directory
    with open(target_dir,"r",encoding=self.encoding) as f:
      self.raw_text = f.read() # raw text
      f.close()
    self.corpus_size = len(self.raw_text)
    self.vocab = sorted(list(set(self.raw_text))) # all the characters in the vocab
    self.vocab_size = len(self.vocab) # length of the vocab
    self.sample_2_index = {ch:i for i,ch in enumerate(self.vocab)} # convert vocab samples to indices
    self.index_2_sample = {i:ch for i,ch in enumerate(self.vocab)} # convert an index to a vocab sample
    self.encode = lambda s: [self.sample_2_index[c] for c in s]
    self.decode = lambda l: "".join([self.index_2_sample[i] for i in l])
    if self.token_mode == "uniform":
      self.uniform_tokenization_mode()
    else:
      self.shifted_tokenization_mode()

  def uniform_tokenization_mode(self):
    text_encoded = self.encode(self.raw_text)
    dataset = list()
    labels = list()  
    for i in range(0,self.corpus_size,self.block_size):
      if(len(text_encoded[i:i+self.block_size]) < self.block_size):
        break
      dataset.append(text_encoded[i:i+self.block_size])
      labels.append(text_encoded[i+self.block_size])
    self.x = torch.tensor(dataset,dtype = torch.long)
    self.y = torch.tensor(labels,dtype = torch.long)

  def shifted_tokenization_mode(self):
    text_encoded = self.encode(self.raw_text)
    dataset = list()
    labels = list()  
    for i in range(0,self.corpus_size,self.block_size):
      if(len(text_encoded[i:i+self.block_size]) < self.block_size):
        break
      dataset.append(text_encoded[i:i+self.block_size])
      labels.append(text_encoded[i+1:i+self.block_size+1])
    self.x = torch.tensor(dataset,dtype = torch.long)
    self.y = torch.tensor(labels,dtype = torch.long)


  def __len__(self):
    return len(self.x)

  def __getitem__(self,index):
    return self.x[index],self.y[index]

        
