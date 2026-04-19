import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk: int):
        super().__init__()
        self.dk = dk

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (B, h, T, dk)
        # mask: (B, 1, 1, T) boolean
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)  # (B, h, T, T)
        output = torch.matmul(weights, V)  # (B, h, T, dk)
        return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dk: int):
        super().__init__()
        self.heads = heads
        self.dk = dk
        self.d_model = d_model
        
        self.W_Q = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(heads)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(heads)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(heads)])
        self.W_O = nn.Linear(heads * dk, d_model)
        self.attn = ScaledDotProductAttention(dk)
        
    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.size()
        head_outputs = []
        head_weights = []
        for h in range(self.heads):
            # Reshape Q, K, V to (B, 1, T, dk) to easily use attention
            Q = self.W_Q[h](x).unsqueeze(1)
            K = self.W_K[h](x).unsqueeze(1)
            V = self.W_V[h](x).unsqueeze(1)
            out, w = self.attn(Q, K, V, mask=mask)
            head_outputs.append(out.squeeze(1)) # (B, T, dk)
            head_weights.append(w.squeeze(1))   # (B, T, T)
            
        # Concatenate
        concat = torch.cat(head_outputs, dim=-1) # (B, T, heads*dk)
        out = self.W_O(concat) # (B, T, d_model)
        # return out and stacked weights (B, heads, T, T)
        return out, torch.stack(head_weights, dim=1)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dk: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, heads, dk)
        self.dropout1 = nn.Dropout(0.1)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Pre-LN
        mha_out, weights = self.mha(self.ln1(x), mask=mask)
        x = x + self.dropout1(mha_out)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out)
        return x, weights

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, heads: int, dk: int, d_ff: int, layers: int, num_classes: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=1000)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, heads, dk, d_ff) for _ in range(layers)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, padding_mask=None, return_attn=False):
        # input_ids: (B, T)
        B, T = input_ids.size()
        x = self.token_embedding(input_ids) # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1) # (B, T+1, d_model)
        x = self.pos_encoding(x)
        
        # padding_mask shape adjustment to match T+1. The CLS token is never masked.
        if padding_mask is not None:
            # padding_mask: (B, T) boolean, True where padding
            # add a False for CLS
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            ext_mask = torch.cat([cls_mask, padding_mask], dim=1)
            # transform to (B, 1, 1, T+1) for attention
            mask = ext_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        all_weights = []
        for block in self.encoder_blocks:
            x, weights = block(x, mask=mask)
            all_weights.append(weights)
            
        cls_repr = x[:, 0, :] # (B, d_model)
        logits = self.mlp_head(cls_repr)
        
        if return_attn:
            return logits, all_weights
        return logits
