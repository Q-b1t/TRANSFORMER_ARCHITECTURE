import os
import re

pattern = r"[3$&]{1,}"

# write the dataset to a specific file

def write_file(name,dataset):
    with open(name,"wb") as f:
        for line in dataset:
            f.write(line.encode())
        print(f"{len(dataset)} were writen into {name}.")
        f.close()

# retrieve file corpus
handler = open("shkspr.txt","r")
text_corpus = list()
for line in handler:
    if  re.match(pattern=pattern,string=line):
        continue
    else:
        text_corpus.append(line)

corpus_size = len(text_corpus)
train_percentage = 0.8
train_samples_number = int(corpus_size * train_percentage)

# split data
train_dataset = text_corpus[:train_samples_number]
validation_dataset = text_corpus[train_samples_number:]

print(f"Total lines: {corpus_size}\nTrain percentage: {train_percentage}\nNumber of train lines: {len(train_dataset)}\nNumber of validation lines: {len(validation_dataset)}")

# write data to different files
train_file_name = "training_data.txt"
validation_file_name = "validation_data.txt"

write_file(train_file_name,train_dataset)
write_file(validation_file_name,validation_dataset)