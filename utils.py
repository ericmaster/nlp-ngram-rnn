#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   
# Copyright (C) 2024
# 
# @author: Ezra Fu <erzhengf@andrew.cmu.edu>
# based on work by 
# Ishita <igoyal@andrew.cmu.edu> 
# Suyash <schavan@andrew.cmu.edu>
# Abhishek <asrivas4@andrew.cmu.edu>

"""
11-411/611 NLP Assignment 2
This file contains the preprocessing and read() functions. Don't edit this file.
"""

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import random
import math
import os
from pathlib import Path

START = "<s>"
EOS = "</s>"
UNK = "<UNK>"

def read_file(file_path):
    """
    Read a single text file.
    """
    with open(file_path, 'r') as f:
        text = f.readlines()
    return text

def preprocess(sentences, n):
    """
    Args:
        sentences: List of sentences
        n: n-gram value

    Returns:
        preprocessed_sentences: List of preprocessed sentences
    """
    sentences = add_special_tokens(sentences, n)

    preprocessed_sentences = []
    for line in sentences:
        preprocessed_sentences.append([tok.lower() for tok in line.split()])
    
    return preprocessed_sentences

def add_special_tokens(sentences, ngram):
    num_of_start_tokens = ngram - 1 if ngram > 1 else 1
    start_tokens = " ".join([START] * num_of_start_tokens)
    sentences = ['{} {} {}'.format(start_tokens, sent, EOS) for sent in sentences]
    return sentences


def flatten(lst):
    """
    Flattens a nested list into a 1D list.
    Args:
        lst: Nested list (2D)
    
    Returns:
        Flattened 1-D list
    """
    return [b for a in lst for b in a]

def loadfile(filename):
    # Read the file
    tokens = flatten(preprocess(read_file(filename), n=1))

    # Build a vocabulary: set of unique words
    vocab = set(tokens)
    vocab.add(START)
    vocab.add(EOS)
    vocab.add(UNK)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    # Convert words to indices
    indices = [word_to_ix[word] for word in tokens]

    # Prepare dataset
    class WordDataset(Dataset):
        def __init__(self, data, sequence_length=30):
            self.data = data
            self.sequence_length = sequence_length

        def __len__(self):
            return len(self.data) - self.sequence_length

        def __getitem__(self, index):
            return (
                torch.tensor(self.data[index:index+self.sequence_length], dtype=torch.long),
                torch.tensor(self.data[index+1:index+self.sequence_length+1], dtype=torch.long),
            )

    dataset = WordDataset(indices, sequence_length=min(30, len(indices)-1))
    return vocab, word_to_ix, ix_to_word, DataLoader(dataset, batch_size=2, shuffle=True)

def best_candidate(lm, prev, i, without=[], mode="random"):
    """
    Returns the most probable word candidate after a given sentence.
    """
    blacklist  = [UNK] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        if(mode=="random"):
            return candidates[random.randrange(len(candidates))]
        else:
            return candidates[0]
        
def top_k_best_candidates(lm, prev, k, without=[]):
    """
    Returns the K most-probable word candidate after a given n-1 gram.
    Args
    ----
    lm: LanguageModel class object
    
    prev: n-1 gram
        List of tokens n
    """
    blacklist  = [UNK] + without
    candidates = ((ngram[-1],prob) for ngram,prob in lm.model.items() if ngram[:-1]==prev)
    candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("</s>", 1)
    else:
        return candidates[:k]
    
def generate_sentences_from_phrase(lm, num, sent, prob, mode):
    """
    Generate sentences using the trained language model.
    """
    min_len=12
    max_len=24
    
    for i in range(num):
        while sent[-1] != "</s>":
            prev = () if lm.n == 1 else tuple(sent[-(lm.n-1):])
            blacklist = sent + (["</s>"] if len(sent) < min_len else [])

            next_token, next_prob = best_candidate(lm, prev, i, without=blacklist, mode=mode)
            sent.append(next_token)
            prob *= next_prob
            
            if len(sent) >= max_len:
                sent.append("</s>")

        yield ' '.join(sent), -1/math.log(prob)
        
def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            embeddings_dict[word] = vector
    return embeddings_dict

def create_embedding_matrix(word_to_ix, embeddings_dict, embedding_dim):
    vocab_size = len(word_to_ix)
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    for word, ix in word_to_ix.items():
        if word in embeddings_dict:
            embedding_matrix[ix] = embeddings_dict[word]
        else:
            embedding_matrix[ix] = torch.rand(embedding_dim)  # Random initialization for words not in GloVe
    return embedding_matrix

def split_and_save_datasets(data_dir="data/lyrics/", train_ratio=0.9):
    """
    Split text files into train and test sets and save them in separate folders.
    
    Args:
        data_dir: Directory containing the source text files
        train_ratio: Ratio of data to use for training (default: 0.9)
    """
    # Create output directories if they don't exist
    train_dir = Path("data/train")
    test_dir = Path("data/test")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all text files except test_lyrics.txt
    files = [f for f in Path(data_dir).glob("*.txt") if f.name != "test_lyrics.txt"]
    
    train_splits = {}
    test_splits = {}
    
    # Process each file
    for f in files:
        # Read and split the file
        data = read_file(f)
        train_len = int(len(data) * train_ratio)
        
        # Store splits in dictionaries
        train_splits[f.name] = data[:train_len]
        test_splits[f.name] = data[train_len:]
        
        # Save train split
        train_path = train_dir / f.name
        with open(train_path, 'w', encoding='utf-8') as train_file:
            train_file.writelines(train_splits[f.name])
            
        # Save test split
        test_path = test_dir / f.name
        with open(test_path, 'w', encoding='utf-8') as test_file:
            test_file.writelines(test_splits[f.name])
    
    return train_splits, test_splits

def calculate_perplexity(model, text, word_to_ix, sequence_length=30):
    """
    Calculate the perplexity of an RNN language model on a given text.
    Uses similar preprocessing and data handling to loadfile().
    
    Args:
        model: The trained RNN language model
        text: Input text string to evaluate
        word_to_ix: Dictionary mapping words to indices
        sequence_length: Length of sequences to process (default: 30)
    
    Returns:
        float: Perplexity score
    """
    model.eval()  # Set model to evaluation mode
    
    # Preprocess using the same approach as in utils
    sentences = [text]  # Treat input text as a single sentence
    preprocessed = preprocess(sentences, n=1)  # Use n=1 as in loadfile
    tokens = flatten(preprocessed)  # Flatten the preprocessed sentences
    
    # Convert words to indices, handling unknown words
    indices = []
    for token in tokens:
        if token in word_to_ix:
            indices.append(word_to_ix[token])
        else:
            indices.append(word_to_ix[UNK])
    
    # Create dataset using the same WordDataset class approach
    class WordDataset(Dataset):
        def __init__(self, data, sequence_length=30):
            self.data = data
            self.sequence_length = sequence_length

        def __len__(self):
            return len(self.data) - self.sequence_length

        def __getitem__(self, index):
            return (
                torch.tensor(self.data[index:index+self.sequence_length], dtype=torch.long),
                torch.tensor(self.data[index+1:index+self.sequence_length+1], dtype=torch.long),
            )
    
    # Create dataset and dataloader
    sequence_length = min(sequence_length, len(indices)-1)
    dataset = WordDataset(indices, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Don't shuffle for evaluation
    
    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        # Process all batches
        for inputs, targets in dataloader:
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            
            # Get model predictions
            output, _ = model(inputs)
            
            # Calculate loss
            loss = criterion(output.view(-1, len(word_to_ix)), targets.view(-1))
            
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    # Handle edge case where there are no tokens
    if total_tokens == 0:
        return float('inf')
    
    # Calculate average negative log likelihood and perplexity
    avg_nll = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    
    return perplexity