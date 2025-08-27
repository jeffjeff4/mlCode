import torch
import torch.nn as nn

# Vocab size 10000, hidden dim 512
V, H, E = 10000, 512, 768

# ----- Factorized Embedding -----
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)  # V x E
        self.proj = nn.Linear(embedding_size, hidden_size, bias=False)  # E x H
    def forward(self, x):
        return self.proj(self.embedding(x))  # [batch, seq_len, H]

# ----- Weight Tying -----
class TiedEmbeddingSoftmax(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # V x H
        # softmax weights will be tied later
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder.weight = self.embedding.weight  # Tying happens here

    def forward(self, x):
        emb = self.embedding(x)          # input
        logits = self.decoder(emb)       # output
        return logits


# Demo
x = torch.randint(0, V, (2, 5))   # batch=2, seq=5

modelFactor = FactorizedEmbedding(V, H, E)
factor_logits = modelFactor(x)
print(factor_logits.shape)  # [2, 5, 10000]

modelTiedW = TiedEmbeddingSoftmax(V, H)
tiedW_logits = modelTiedW(x)
print(tiedW_logits.shape)  # [2, 5, 10000]




#---------------------------------------------------------------------

import torch
import torch.nn as nn


class LMWithWeightTying(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # V x H
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        # ---- Weight tying happens here ----
        self.decoder.weight = self.embedding.weight

    def forward(self, x):
        emb = self.embedding(x)         # [batch, seq, H]
        out, _ = self.rnn(emb)          # [batch, seq, H]
        logits = self.decoder(out)      # [batch, seq, V]
        return logits

# Demo
x = torch.randint(0, V, (2, 5))   # batch=2, seq=5
model_lmw = LMWithWeightTying(V, H)
lmw_logits = model_lmw(x)
print(lmw_logits.shape)  # [2, 5, 10000]



####🔹 1. Factorized Embedding Parameterization (FEP)
####
####	1. What it does:
####	Factorizes the big 𝑉×𝐻 embedding matrix into two smaller matrices:
####			𝐸=𝐸1⋅𝐸2
####	​
####	𝐸1∈𝑅^(𝑉×𝐸) (token → low-dim embedding)
####	𝐸2∈𝑅^(𝐸×𝐻) (projection into hidden size)
####
####	2. Main benefit: Saves parameters when vocab size is large.
####
####	3. Used in: ALBERT (2019)
####
####	4. Parameters:
####	Original =𝑉×𝐻
####	Factorized = 𝑉×𝐸+𝐸×𝐻
####	where 𝐸≪𝐻.
####
####🔹 2. Weight Tying
####
####	1. What it does:
####		Shares the input embedding matrix and the output projection matrix in the softmax layer.
####
####		In a standard Transformer LM:
####
####		Input: token id → embedding 𝐸∈𝑅^(𝑉×𝐻)
####		Output: hidden state → logits with 𝑊∈𝑅^(𝐻×𝑉)
####
####		Normally, 𝐸 and 𝑊 are independent.
####
####		With weight tying:
####			𝑊=𝐸⊤
####
####		so the same parameters are reused.
####
####	2. Main benefit:
####	Reduces parameters.
####	Encourages consistency: the word vector used in input is the same one used to predict the word at output.
####
####	3. Used in: GPT, BERT, Transformer-XL, etc.
####
####	4. Parameters:
####	Original = 𝑉×𝐻+𝐻×𝑉≈2𝑉𝐻
####	Weight tying = 𝑉×𝐻
####
####🔹 Side-by-Side Comparison
####Aspect						FEP (Factorized Embedding)						Weight Tying
####Where applied			Input embedding layer only						Input & output layers (LM head)
####How it works			Factorizes embedding into 𝐸1⋅𝐸2   				Shares weights between embedding & softmax projection
####Main goal				Reduce parameter count from𝑉×𝐻 →𝑉×𝐸+𝐸×𝐻			Reduce duplication of parameters (2VH → 𝑉*𝐻)
####Saves parameters?		Yes, especially when vocab is large & 𝐸≪𝐻		Yes, cuts nearly in half
####Effect on model			Keeps embeddings low-dim, projected later		Makes input/output word representations consistent
####Used in					ALBERT											GPT, BERT, Transformer-XL, many LMs