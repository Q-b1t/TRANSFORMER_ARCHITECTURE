# GPT Architecture
As the number says, this are the multiple versions of my implementations of the Gradient Pretrained Transformer Architecture. The filesystem of the repository is as follows (I only marked the most recent/updated architecture/custom dataset version):
```
.
└── GPT_ARCHITECTURE/
    ├── data/
    │   ├── split.py
    │   └── *.txt
    ├── DATASET/
    │   └── gpt_dataset_mkIII.py
    ├── GPT_MKI/
    │   └── gpt_architecture_mk_VI.py
    └── GPT_MKII/
        └── gpt_architecture_mkII.py
```
- ```data.py```: Use this directory's ```split.py``` to split the text samples into training and test datasets (stored in text files as well).
- ```GPT_MKI```: My first version of the implementation. Good enough but suboptimal.
- ```GPT_MKII```: My latest version of the implementation. Currently developing/debugging (not deployed yet).
- ```DATASET```: A custom dataset that tokenizes and preprocesses text sequences for my architectures. The latest one now supports sequence padding.

