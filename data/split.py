#!/usr/bin/python3

import os
import re
import math


class FILE_DOES_NOT_EXIST(Exception):
    # raise if no input file is supplied
    pass


pattern = r"[3$&]{1,}"

# write the dataset to a specific file

def write_file(name,dataset):
    with open(name,"wb") as f:
        for line in dataset:
            f.write(line.encode())
        print(f"{len(dataset)} were writen into {name}.")
        f.close()


# retrieve file corpus
file_name = str(input("Enter the target text file:\n"))
dataset_percentage = float(input("Enter the percentage [0.0:1.0] of the text file you want to use to create the dataset (recomended: 0.9):\n"))
train_percentage = float(input("Enter the train-test split [0.0:1.0] (recomended: 0.8):\n"))



try:
    handler = open(file_name,"r")
except:
    raise FILE_DOES_NOT_EXIST("The file could not be found.")
    exit()

text_corpus = list()
for line in handler:
    if  re.match(pattern=pattern,string=line):
        continue
    else:
        text_corpus.append(line)


# total number of lines in the dataset
corpus_size = len(text_corpus)

"""
important parameters: change the dataset_percentage and the train_percentage
to determine the amount of data that will be taken from the dataset for training
"""

validation_percentage = math.ceil(1 - train_percentage)
dataset_length = int(corpus_size * dataset_percentage)
train_samples_number = int(dataset_length * train_percentage)
validation_samples_number = int(dataset_length * validation_percentage)

# split data
train_dataset = text_corpus[:train_samples_number]
validation_dataset = text_corpus[train_samples_number:validation_samples_number]

print(f"Total lines: {dataset_length}\nTrain percentage: {train_percentage}\nNumber of train lines: {len(train_dataset)}\nNumber of validation lines: {len(validation_dataset)}")

# write data to different files
train_file_name = "training_data.txt"
validation_file_name = "validation_data.txt"

write_file(train_file_name,train_dataset)
write_file(validation_file_name,validation_dataset)

