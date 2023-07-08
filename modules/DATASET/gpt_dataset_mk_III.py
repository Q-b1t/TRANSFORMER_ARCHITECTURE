import torch
import torch.nn as nn
from torch.utils.data import Dataset

torch.manual_seed(4444)

"""
This class is a modest update from the GptDatasetMKII class, and incorporates token padding (<SOS> and <EOS> tokens denoting the start and end of sequences).
"""

class GptDatasetMKIII(Dataset):
  """
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
  padding_tokens: Expected a dictionary mapping key strings '<SOS>' and '<EOS>' to integer values in order to have padding tokens
  to denote the start and end of sequences repectively. Since the token-index mappings use incremental sequential natural numbers,
  it is recomended to use negative values for the padding tokens of use values greater than the expected vocab size.
  """
  def __init__(self,target_dir,encoding = "utf-8",block_size = 8,tokenization_mode = "shifted",padding_tokens = {"<SOS>":-1,"<EOS>":-2},vocab = None):
    # instance important parameters
    self.text_path = target_dir # directory containing the text corpus
    self.encoding = encoding # encoding used to read the text
    self.block_size = block_size # length the sequence to tokenize and parse the samples
    self.token_mode = tokenization_mode # can be "uniform" or "shifted"

    # instance padding tokens if they are receibed
    if padding_tokens is not None:
      assert self.block_size > 3,"If padding is enabled, the block size must be at least of size 3 -> [<SOS>,token,<EOS>]"
      self.padding_tokens = padding_tokens
      self.padding_mode = True
    else:
      self.padding_tokens = None
      self.padding_mode = False

    # read and preprocess the dataset
    self.retrieve_raw_text()

    
    self.vocab = sorted(list(set(self.raw_text))) if vocab is None else vocab # all the characters in the vocab
    self.vocab.append("<UKN>") # append a special character for unkown characters
      

    # create the index mappings
    self.create_index_mappings(padding=self.padding_mode)

    self.vocab_size = len(self.vocab) # length of the vocab

    if self.token_mode == "uniform":
      self.uniform_tokenization_mode(padding=self.padding_mode)
    else:
      self.shifted_tokenization_mode(padding=self.padding_mode)

  def retrieve_raw_text(self):
    """
    Retrieves the text from the provided path to the sample text file.
    It saves the text, vocabulary, and lengths as class attributes.
    """
    # retrieve the text corpus from the target directory
    with open(self.text_path,"r",encoding=self.encoding) as f:
      self.raw_text = f.read() # raw text
      f.close()
    # save useful parameters as class attributes
    self.corpus_size = len(self.raw_text)


  def create_index_mappings(self,padding):
    """
    Creates the token to index and index to token mappings as well as their respective encoding and decoding functions
    """
    self.sample_2_index = {ch:i for i,ch in enumerate(self.vocab)} # convert vocab samples to indices
    self.index_2_sample = {i:ch for i,ch in enumerate(self.vocab)} # convert an index to a vocab sample

    if padding:
      self.reverse_tokens = {value : key for key, value in self.padding_tokens.items()} # create the reverse version of the padding tokens
      # update the dictionaries with the <SOS> and <EOS> mappings
      self.sample_2_index = self.sample_2_index | self.padding_tokens
      self.index_2_sample = self.index_2_sample | self.reverse_tokens

    # create the respective encoding and decoding functions
    self.encode = lambda s: [self.sample_2_index[c] if c in self.sample_2_index.keys() else self.sample_2_index["<UKN>"] for c in s]
    self.decode = lambda l: "".join([self.index_2_sample[i] for i in l])

  def uniform_tokenization_mode(self,padding):
    text_encoded = self.encode(self.raw_text)
    dataset = list()
    labels = list()
    stepsize = self.block_size if not padding else self.block_size -2
    for i in range(0,self.corpus_size,stepsize):
      if((i+self.block_size > self.corpus_size) or (padding and len(text_encoded[i:i+stepsize]) + 2  < self.block_size)  or (not padding and len(text_encoded[i:i+stepsize]) < self.block_size) ):
        break
      else:
        if padding:
          dataset.append([self.sample_2_index["<SOS>"]]+ text_encoded[i:i+stepsize] + [self.sample_2_index["<EOS>"]])
          labels.append([self.sample_2_index["<SOS>"]]+  [text_encoded[i+stepsize]] + [self.sample_2_index["<EOS>"]])
        else:
          dataset.append(text_encoded[i:i+stepsize])
          labels.append(text_encoded[i+stepsize])
    try:
      self.x = torch.tensor(dataset,dtype = torch.long)
      self.y = torch.tensor(labels,dtype = torch.long)
    except:
      dataset = dataset[:-1]
      labels = dataset[:-1]
      self.x = torch.tensor(dataset,dtype = torch.long)
      self.y = torch.tensor(labels,dtype = torch.long)

  def shifted_tokenization_mode(self,padding):
    text_encoded = self.encode(self.raw_text)
    dataset = list()
    labels = list()
    stepsize = self.block_size if not padding else self.block_size -2

    for i in range(0,self.corpus_size,stepsize):
      if( (i+self.block_size > self.corpus_size) or (padding and len(text_encoded[i:i+stepsize]) + 2  < self.block_size)  or (not padding and len(text_encoded[i:i+stepsize]) < self.block_size) ):
        break
      else:
        if padding:
          dataset.append([self.sample_2_index["<SOS>"]]+ text_encoded[i:i+stepsize] + [self.sample_2_index["<EOS>"]])
          labels.append([self.sample_2_index["<SOS>"]]+  text_encoded[i+1:i+stepsize+1] + [self.sample_2_index["<EOS>"]])
        else:
          dataset.append(text_encoded[i:i+stepsize])
          labels.append(text_encoded[i+1:i+stepsize+1])

    try:
      self.x = torch.tensor(dataset,dtype = torch.long)
      self.y = torch.tensor(labels,dtype = torch.long)
    except:
      dataset = dataset[:-1]
      labels = dataset[:-1]
      self.x = torch.tensor(dataset,dtype = torch.long)
      self.y = torch.tensor(labels,dtype = torch.long)

  def __len__(self):
    return len(self.x)

  def __getitem__(self,index):
    return self.x[index],self.y[index]