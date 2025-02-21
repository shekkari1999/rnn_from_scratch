# data.py
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re
from collections import Counter
import config

# Load Wikitext-2 dataset
def load_wikitext():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train, test, val = dataset['train']['text'], dataset['test']['text'], dataset['validation']['text']
    return train, test, val

# Clean text
def clean_text(text):
    text = re.sub(r"@\s*-\s*@", "-", text)  # Fix hyphenated words
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    text = re.sub(r"\[[^]]*\]", "", text)  # Remove bracketed text
    return text

# Tokenizer (Create Vocab)
def create_vocab(data, min_freq=config.MIN_FREQ):
    all_text = " ".join(data)
    words = all_text.split()
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    
    stoi = {word: i+2 for i, word in enumerate(vocab)}  # Reserve 0, 1 for special tokens
    stoi['<PAD>'] = 0
    stoi['<UNK>'] = 1
    itos = {i: word for word, i in stoi.items()}
    
    return stoi, itos

# Convert sentences to token indices
def convert_sentences_to_tokens(data, stoi):
    datas = torch.full((len(data), config.MAX_LENGTH), stoi['<PAD>'], dtype=torch.long)

    for i, line in enumerate(data):
        tokenized = [stoi.get(word, stoi['<UNK>']) for word in line.split()]
        tokenized = tokenized[:config.MAX_LENGTH]  # Truncate
        datas[i, :len(tokenized)] = torch.tensor(tokenized, dtype=torch.long)

    return datas

# Define PyTorch Dataset class
class TextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data.clone().detach()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        X = sentence[:-1]  # Input
        y = sentence[1:]   # Target
        return X, y

# Function to prepare dataset & DataLoader
def prepare_data():
    train, test, val = load_wikitext()
    
    # Clean text
    train = [clean_text(line) for line in train if line.strip()]
    test = [clean_text(line) for line in test if line.strip()]
    val = [clean_text(line) for line in val if line.strip()]

    stoi, itos = create_vocab(train + val + test)
    
    train_tokens = convert_sentences_to_tokens(train, stoi)
    val_tokens = convert_sentences_to_tokens(val, stoi)
    test_tokens = convert_sentences_to_tokens(test, stoi)

    train_data = TextDataset(train_tokens)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)

    return train_loader, stoi, itos