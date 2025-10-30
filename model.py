import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super.__init__()
        self.d_model: int = d_model
        self.vocab_size: int = vocab_size
        self.embedding = nn.Embedding(num_embeddings= vocab_size, embedding_dim= d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.seq_len: int = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape(1), :]).requires_grad_(False)
        return self.dropout(x)
    
    
    class LayerNormalization(nn.Module):
        
        def __init__(self, eps: float = 10**-6) -> None:
            super().__init__()
            self.esp: float = eps
            self.alpha = nn.Parameter(torch.ones(1)) # Multipled
            self.bias = nn.Parameter(torch.zeros(1)) # Added
            
        def forward(self, x: torch.Tensor):
            mean = x.mean(dim= -1, keepdim=True)
            std = x.std(dim= -1, keepdim=True)
            return self.alpha * (x - mean) / (std + self.esp) + self.bias
        
        
    class FeedForwardBlock(nn.Module):
        
        def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
            super().__init__()
            self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
            self.dropout = nn.Dropout(dropout)
            
            self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
            
        def forward(self, x):
            # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
            
            return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        
        
        