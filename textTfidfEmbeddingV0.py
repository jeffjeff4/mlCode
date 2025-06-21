##import torch
##import torch.nn as nn
##import torch.optim as optim
##from collections import Counter
##import numpy as np
##
### Sample corpus
##corpus = [
##    "word embeddings are cool",
##    "word2vec is a popular embedding model",
##    "pytorch makes deep learning easy"
##]
##
### Hyperparameters
##EMBEDDING_DIM = 100
##WINDOW_SIZE = 2
##BATCH_SIZE = 32
##EPOCHS = 100
##
##
### Preprocessing
##def preprocess(corpus):
##    words = []
##    for sentence in corpus:
##        words.extend(sentence.lower().split())
##    vocab = Counter(words)
##    vocab = sorted(vocab, key=vocab.get, reverse=True)
##    word2idx = {word: i for i, word in enumerate(vocab)}
##    idx2word = {i: word for i, word in enumerate(vocab)}
##    return word2idx, idx2word, vocab
##
##
##word2idx, idx2word, vocab = preprocess(corpus)
##VOCAB_SIZE = len(vocab)
##
##
### Generate training pairs
##def create_training_data(corpus, word2idx, window_size):
##    training_data = []
##    for sentence in corpus:
##        sentence = sentence.lower().split()
##        for i, target_word in enumerate(sentence):
##            for j in range(i - window_size, i + window_size + 1):
##                if j != i and 0 <= j < len(sentence):
##                    context_word = sentence[j]
##                    training_data.append((word2idx[target_word], word2idx[context_word]))
##    return training_data
##
##
##training_data = create_training_data(corpus, word2idx, WINDOW_SIZE)
##
##
### Word2Vec model
##class Word2Vec(nn.Module):
##    def __init__(self, vocab_size, embedding_dim):
##        super(Word2Vec, self).__init__()
##        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
##        self.linear = nn.Linear(embedding_dim, vocab_size)
##
##    def forward(self, x):
##        x = self.embeddings(x)
##        x = self.linear(x)
##        return x
##
##
##model = Word2Vec(VOCAB_SIZE, EMBEDDING_DIM)
##criterion = nn.CrossEntropyLoss()
##optimizer = optim.Adam(model.parameters(), lr=0.001)
##
### Training loop
##for epoch in range(EPOCHS):
##    total_loss = 0
##    for target, context in training_data:
##        target_tensor = torch.tensor([target], dtype=torch.long)
##        context_tensor = torch.tensor([context], dtype=torch.long)
##
##        optimizer.zero_grad()
##        output = model(target_tensor)
##        loss = criterion(output, context_tensor)
##        loss.backward()
##        optimizer.step()
##
##        total_loss += loss.item()
##
##    if (epoch + 1) % 10 == 0:
##        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(training_data):.4f}')
##
### Get word embeddings
##embeddings = model.embeddings.weight.data
##print(f"Embedding for 'word': {embeddings[word2idx['word']][:10]}")  # First 10 dimensions
##
##import gensim.downloader as api
##from torch.utils.data import Dataset, DataLoader
##
### Load pre-trained model
##w2v_model = api.load('word2vec-google-news-300')
##
##
### Create PyTorch embedding layer
##class PretrainedEmbeddingLayer(nn.Module):
##    def __init__(self, word2vec_model, freeze=True):
##        super().__init__()
##        vocab_size = len(word2vec_model.key_to_index)
##        embedding_dim = word2vec_model.vector_size
##
##        # Create embedding matrix
##        embedding_matrix = torch.zeros((vocab_size, embedding_dim))
##        for i, word in enumerate(word2vec_model.key_to_index):
##            embedding_matrix[i] = torch.tensor(word2vec_model[word])
##
##        self.embedding = nn.Embedding.from_pretrained(
##            embedding_matrix,
##            freeze=freeze,
##            padding_idx=0
##        )
##
##    def forward(self, x):
##        return self.embedding(x)
##
##
### Example usage
##embedding_layer = PretrainedEmbeddingLayer(w2v_model)
##print(f"Embedding layer shape: {embedding_layer.embedding.weight.shape}")
##
##
##from transformers import BertModel, BertTokenizer
##import torch
##
### Initialize tokenizer and model
##tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
##model = BertModel.from_pretrained('bert-base-uncased')
##
### Input text
##text = "Natural language processing with PyTorch is powerful"
##
### Tokenize and prepare input
##inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
##
### Get embeddings
##with torch.no_grad():
##    outputs = model(**inputs)
##
### Last hidden state (sequence embeddings)
##last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
##print(f"BERT embedding shape: {last_hidden_state.shape}")
##
### Pooled output (sentence embedding)
##pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]
##print(f"Pooled output shape: {pooled_output.shape}")
##
### Get specific token embedding
##token_embeddings = last_hidden_state[0]  # First sequence
##print(f"Embedding for 'language': {token_embeddings[2][:10]}")  # First 10 dimensions


