import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class LanguageModel(nn.Module):
    def __init__(self,
                 model,
                 embedding_dim,
                 vocab_size):
        super().__init__()

        self.model = model
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        init.xavier_uniform_(self.fc_out.weight.data)
        self.fc_out.bias.data.fill_(0)

    def forward(self, tokens, mask=None):

        #tokens = [seq len, batch size]

        if mask is None:
            embedded = self.model(tokens)
        else:
            embedded = self.model(tokens, mask)

        #embedded = [batch size, embedding dim, seq len]

        embedded = embedded.permute(0, 2, 1)

        #embedded = [batch size, seq len, embedding dim]

        predictions = self.fc_out(embedded)

        #predictions = [batch size, seq len, vocab size]

        return predictions

class EmbeddingPooler(nn.Module):
    def __init__(self,
                 emb_dim,
                 pool_mode):
        super().__init__()

        assert pool_mode in ['max', 'weighted_mean']

        self.pool_mode = pool_mode

        if pool_mode == 'weighted_mean':
            self.fc = nn.Linear(emb_dim, 1, bias = False)

    def forward(self, 
                embeddings,
                mask):

        #embeddings = [batch, emb dim, seq len]
        #mask = [batch, seq len]

        _, _, seq_len = embeddings.shape

        mask = mask.unsqueeze(1)

        #mask = [batch, 1, seq len]

        if self.pool_mode == 'max':
            embeddings = embeddings.masked_fill(mask == 0, -1e10)
            pooled = F.max_pool1d(embeddings,
                                  kernel_size = seq_len)

        elif self.pool_mode == 'weighted_mean':
            _embeddings = embeddings.permute(0, 2, 1)
            #_embeddings = [batch, seq len, emb dim]
            weights = torch.sigmoid(self.fc(_embeddings))
            #weighs = [batch, seq len, 1]
            weights = weights.permute(0, 2, 1)
            #weights = [batch, 1, seq len]
            weighted = embeddings * weights
            weighted = weighted.masked_fill(mask == 0, 0)
            #weighted = [batch, emb dim, seq len]
            pooled = F.avg_pool1d(weighted,
                                  kernel_size = seq_len)

        else:
            raise ValueError(f'Unknown pool mode: {self.pool_mode}')

        #pooled = [batch size, emb dim, 1]

        pooled = pooled.squeeze(-1)

        #pooled = [batch size, emb dim]

        return pooled

class EmbeddingPredictor(nn.Module):
    def __init__(self,
                 emb_dim,
                 output_dim,
                 pool_mode):
        super().__init__()

        assert pool_mode in ['max', 'weighted_mean']

        self.pool_mode = pool_mode

        if pool_mode == 'weighted_mean':
            self.fc = nn.Linear(emb_dim, 1, bias = False)

        self.fc_out = nn.Linear(emb_dim, output_dim)

    def forward(self, 
                embeddings,
                mask):

        #embeddings = [batch, emb dim, seq len]
        #mask = [batch, seq len]

        _, _, seq_len = embeddings.shape

        mask = mask.unsqueeze(1)

        #mask = [batch, 1, seq len]

        if self.pool_mode == 'max':
            embeddings = embeddings.masked_fill(mask == 0, -1e10)
            pooled = F.max_pool1d(embeddings,
                                  kernel_size = seq_len)

        elif self.pool_mode == 'weighted_mean':
            _embeddings = embeddings.permute(0, 2, 1)
            #_embeddings = [batch, seq len, emb dim]
            weights = torch.sigmoid(self.fc(_embeddings))
            #weighs = [batch, seq len, 1]
            weights = weights.permute(0, 2, 1)
            #weights = [batch, 1, seq len]
            weighted = embeddings * weights
            weighted = weighted.masked_fill(mask == 0, 0)
            #weighted = [batch, emb dim, seq len]
            pooled = F.avg_pool1d(weighted,
                                  kernel_size = seq_len)

        else:
            raise ValueError(f'Unknown pool mode: {self.pool_mode}')

        #pooled = [batch size, emb dim, 1]

        pooled = pooled.squeeze(-1)

        #pooled = [batch size, emb dim]

        prediction = self.fc_out(pooled)

        #prediction = [batch size, output dim]

        return prediction

class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 dropout,
                 pad_idx,
                 device):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(250, emb_dim)

        self.layers = nn.ModuleList([TransformerEncoderLayer(emb_dim, hid_dim, n_heads, dropout) for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(emb_dim, 1e-3)
        self.dropout = nn.Dropout(dropout)

        self.pad_idx = pad_idx
        self.device = device

    def forward(self,
                tokens,
                mask = None
                ):

        #tokens = [seq len, batch size]

        tokens = tokens.permute(1, 0)

        #tokens = [batch size, seq len]

        if mask is None:
            mask = (tokens != self.pad_idx).unsqueeze(1).unsqueeze(2)

        #mask = [batch size, 1, 1, seq len]

        pos = torch.arange(0, tokens.shape[1]).unsqueeze(0).repeat(tokens.shape[0], 1).to(self.device)

        #pos = [batch size, seq len]

        tok_embedded = self.tok_embedding(tokens)
        pos_embedded = self.pos_embedding(pos)

        #tok/pos_embedded = [batch size, seq len, emb dim]

        embedded = self.dropout(self.layer_norm(tok_embedded + pos_embedded))

        #embedded = [batch size, seq len, emb dim]

        for layer in self.layers:
            embedded = layer(embedded, mask)

        #embedded = [batch size, seq len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        #embedded = [batch size, emb dim, seq len]

        return embedded

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 emb_dim,
                 hid_dim,
                 n_heads,
                 dropout):
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(emb_dim, eps=1e-3)
        self.layer_norm_2 = nn.LayerNorm(emb_dim, eps=1e-3)
        self.self_attn = MultiHeadAttention(emb_dim, n_heads, dropout)
        self.fc_1 = nn.Linear(emb_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                embedded,
                mask):

        #embedded = [batch size, seq len, emb dim]

        embedded = self.layer_norm_1(embedded + self.dropout(self.self_attn(embedded, embedded, embedded, mask)))

        #embedded = [batch size, seq len, emb dim]

        embedded = self.layer_norm_2(embedded + self.dropout(self.fc_2(self.dropout(F.gelu(self.fc_1(embedded))))))

        #embedded = [batch size, seq len, emb dim]

        return embedded

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 emb_dim,
                 n_heads,
                 dropout):
        super().__init__()

        assert emb_dim % n_heads == 0

        self.fc_q = nn.Linear(emb_dim, emb_dim)
        self.fc_k = nn.Linear(emb_dim, emb_dim)
        self.fc_v = nn.Linear(emb_dim, emb_dim)

        self.fc = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.emb_dim = emb_dim

    def forward(self,
                query,
                key,
                value,
                mask = None):

        #query/key/value = [batch size, seq len, emb dim]

        batch_size = query.shape[0]

        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        #q/k/v = [batch size, seq len, emb dim]

        q = q.view(batch_size, -1, self.n_heads, self.emb_dim // self.n_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.emb_dim // self.n_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.emb_dim // self.n_heads).permute(0, 2, 1, 3)

        #q/k/v = [batch size, n heads, seq len, hid dim // n heads]

        energy = torch.matmul(q, k.permute(0, 1, 3, 2))

        #energy = [batch size, n heads, seq len, seq len]

        energy = energy * (float(self.emb_dim // self.n_heads) ** -0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(energy, dim = -1))

        #attention = [batch size, n heads, seq len, seq len]

        weighted = torch.matmul(attention, v)

        #weighted = [batch size, n heads, seq len, emb dim // n heads]

        weighted = weighted.permute(0, 2, 1, 3).contiguous()

        #weighted = [batch size, n heads, seq len, emb dim // n heads]

        weighted = weighted.view(batch_size, -1, self.n_heads * (self.emb_dim // self.n_heads))

        #weighted = [batch size, seq len, emb dim]
        
        weighted = self.fc(weighted)

        #weighted = [batch size, seq len, emb dim]

        return weighted