import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.query = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.n_heads = n_heads
        self.head_size = head_size
        self.projection = nn.Linear(n_embed, n_embed)
    
    @staticmethod
    def attention(q, k, v, mask=None):
        d_k = k.shape[-1]
        
        dot_product = torch.matmul(q, k.transpose(-2, -1)) # (batch_size, n_heads, seq_len, seq_len)
        scaled_dot_product = dot_product * d_k ** -0.5
        attention_scores = scaled_dot_product    
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        
        attention_scores = F.softmax(attention_scores, dim=-1) # (batch_size, n_heads, seq_len, seq_len)
        out = torch.matmul(attention_scores, v) # (batch_size, n_heads, seq_len, head_size)
        return out
        
    def forward(self, q, k, v, mask=None):
        query = self.query(q) # (batch_size, seq_len, d_model)
        key = self.key(k) # (batch_size, seq_len, d_model)
        value = self.value(v) # (batch_size, seq_len, d_model)
        
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, n_heads, head_size) --> (batch_size, n_heads, seq_len, head_size)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.head_size).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.head_size).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.head_size).transpose(1, 2)
        
        # (batch_size, n_heads, seq_len, head_size)
        x = MultiHeadAttention.attention(query, key, value, mask) 
        # (batch_size, n_heads, seq_len, head_size) --> (batch_size, seq_len, n_heads * head_size == d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.head_size)
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        x = self.projection(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout_p):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(4 * n_embed, n_embed))
    
    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_heads, dropout_p):
        super(EncoderBlock, self).__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_embed, n_heads, head_size)
        self.ffwd = FeedForward(n_embed, dropout_p) 
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x, mask=None):
        # Residual connections
        x = self.norm1(x + self.dropout(self.sa(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffwd(x)))
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embed, max_length, n_heads, n_layers, dropout_p, device):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(max_length, n_embed)
        self.blocks = nn.ModuleList([
            EncoderBlock(n_embed, n_heads, dropout_p) for _ in range(n_layers)
        ])       
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, x, mask=None):
        B, T = x.shape
        token_embed = self.token_embedding(x)
        positional_embed = self.pos_embedding(torch.arange(T))
        embedding = self.dropout(token_embed + positional_embed)

        x = embedding
        for block in self.blocks:
            x = block(x, mask)
        return x
    
class DecoderBlock(nn.Module):
    """Decoder Block"""
    def __init__(self, n_embed, max_length, n_heads, dropout_p):
        super(DecoderBlock, self).__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_embed, n_heads, head_size) # masked self-attention 
        self.encoder_decoder_attention = MultiHeadAttention(n_embed, n_heads, head_size)
        self.ffwd = FeedForward(n_embed, dropout_p)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.norm3 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, encoder_output, decoder_mask, padding_mask=None):
        x = self.norm1(x + self.dropout(self.sa(x, x, x, decoder_mask)))
        self.encoder_decoder_attention(x, encoder_output, encoder_output, padding_mask)
        x = self.norm2(x + self.dropout(self.encoder_decoder_attention(x, encoder_output, encoder_output, padding_mask)))
        x = self.norm3(x + self.dropout(self.ffwd(x)))
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embed, max_length, n_heads, n_layers, dropout_p, device):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(max_length, n_embed)
        self.blocks = nn.ModuleList([
            DecoderBlock(n_embed, max_length, n_heads, dropout_p) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, x, enc_out, decoder_mask, padding_mask=None):
        B, T = x.shape
        token_embed = self.token_embedding(x)
        pos_embed = self.pos_embedding(torch.arange(T))
        embedding = self.dropout(token_embed + pos_embed)

        x = embedding
        for block in self.blocks:
            x = block(x, enc_out, decoder_mask, padding_mask)

        x = self.lm_head(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trgt_vocab_size, n_embed, max_length, n_heads, n_layers, dropout_p, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, n_embed, max_length, n_heads, n_layers, dropout_p, device)
        self.decoder = Decoder(trgt_vocab_size, n_embed, max_length, n_heads, n_layers, dropout_p, device)
        self.device = device

    def forward(self, x, y, padding_mask, decoder_mask):
        encoder_output = self.encoder(x, padding_mask)
        out = self.decoder(y, encoder_output, decoder_mask, padding_mask)
        return out