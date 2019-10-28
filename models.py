import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingPooler(nn.Module):
    def __init__(self,
                 emb_dim,
                 pool_mode):
        super().__init__()

        assert pool_mode in ['mean', 'max', 'weighted_mean']

        self.pool_mode = pool_mode

        if pool_mode == 'weighted_mean':
            self.fc = nn.Linear(emb_dim, 1, bias = False)

    def forward(self, 
                embeddings,
                mask):

        #embeddings = [batch, emb dim, seq_len]
        #mask = [batch, seq len]

        _, _, seq_len = embeddings.shape

        mask = mask.unsqueeze(1)

        #mask = [batch, 1, seq len]

        if self.pool_mode == 'mean':
            embeddings = embeddings.masked_fill(mask == 0, 0)
            pooled = F.avg_pool1d(embeddings,
                                  kernel_size = seq_len)

        elif self.pool_mode == 'max':
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
            #weighted = [batch, emb dim, seq_len]
            pooled = F.avg_pool1d(weighted,
                                  kernel_size = seq_len)

        else:
            raise ValueError(f'Unknown pool mode: {self.pool_mode}')

        #pooled = [batch size, emb dim, 1]

        pooled = pooled.squeeze(-1)

        #pooled = [batch size, emb dim]

        return pooled

class BagOfWordsEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tokens):

        #tokens = [seq len, batch size]

        embedded = self.dropout(self.embedding(tokens))

        #embedded = [seq len, batch size, emb dim]

        embedded = embedded.permute(1, 2, 0)

        #embedded = [batch, emb dim, seq len]

        return embedded

class RNNEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 bidirectional,
                 dropout,
                 rnn_type):
        super().__init__()

        assert rnn_type in ['lstm', 'gru']

        self.embedding = nn.Embedding(input_dim, emb_dim)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, 
                              hid_dim, 
                              num_layers = n_layers, 
                              bidirectional = bidirectional,
                              dropout = 0 if n_layers < 2 else dropout)
        else:
            self.rnn = nn.LSTM(emb_dim,
                               hid_dim, 
                               num_layers = n_layers, 
                               bidirectional = bidirectional,
                               dropout = 0 if n_layers < 2 else dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tokens):

        #tokens = [seq len, batch size]

        embedded = self.dropout(self.embedding(tokens))

        #embedded = [seq len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        #outputs = [seq len, batch size, hid dim * 2 if bidirectional else hid dim]
        #if rnn is gru:
        #    hidden = [n layers * 2 if bidirectional else n layers, batch size, hid dim]
        #if rnn is lstm:
        #    hidden is a tuple of (hidden, cell) both with the sizes of the gru hidden above

        outputs = outputs.permute(1, 2, 0)

        #outputs = [batch size, hid dim * 2 if directional else hid dim, seq len]

        return outputs

class CNNEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 filter_size,
                 n_layers,
                 dropout,
                 device):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(250, emb_dim)

        filter_sizes = [filter_size for _ in range(n_layers)]

        self.convs = nn.ModuleList([nn.Conv1d(in_channels = emb_dim, out_channels = emb_dim, kernel_size = fs) for fs in filter_sizes])

        self.dropout = nn.Dropout(dropout)

        self.filter_size = filter_size
        self.device = device

    def forward(self,
                tokens):

        #tokens = [seq len, batch size]

        tokens = tokens.permute(1, 0)

        #tokens = [batch size, seq len]

        pos = torch.arange(0, tokens.shape[1]).unsqueeze(0).repeat(tokens.shape[0], 1).to(self.device)

        tok_embedded = self.tok_embedding(tokens)
        pos_embedded = self.pos_embedding(pos)

        embedded = self.dropout(tok_embedded + pos_embedded)

        #embedded = [batch size, seq len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        #embedded = [batch size, emb dim, seq len]

        for i, conv in enumerate(self.convs):

            next_embedded = conv(embedded)

            #next_embedded = [batch size, emb dim, seq len - filter_size + 1]

            to_pad = tokens.shape[-1] - next_embedded.shape[-1]

            left_pad = to_pad // 2
            right_pad = to_pad - left_pad

            left_padding = torch.zeros_like(next_embedded)[:,:,:left_pad].to(self.device)
            right_padding = torch.zeros_like(next_embedded)[:,:,:right_pad].to(self.device)

            next_embedded = torch.cat((left_padding, next_embedded, right_padding), dim = -1)

            #next_embedded = [batch size, emb dim, seq len]

            embedded = self.dropout(torch.tanh(next_embedded + embedded))

        return embedded

if __name__ == '__main__':
    
    with torch.no_grad():

        vocab_size = 100
        emb_dim = 64
        batch_size = 32
        seq_len = 10

        print('bow model')

        bow_model = BagOfWordsEncoder(vocab_size, emb_dim)

        pool_embedder_mean = EmbeddingPooler(emb_dim, 'mean')
        pool_embedder_max = EmbeddingPooler(emb_dim, 'max')
        pool_embedder_weighted_mean = EmbeddingPooler(emb_dim, 'weighted_mean')

        tokens = torch.randint(vocab_size, (seq_len, batch_size))

        embeddings = bow_model(tokens)
        
        print(tokens.shape)
        print(embeddings.shape)
        print(pool_embedder_mean(embeddings).shape)
        print(pool_embedder_max(embeddings).shape)
        print(pool_embedder_weighted_mean(embeddings).shape)

        hid_dim = 64
        n_layers = 1
        bidirectional = False
        dropout = 0.5
        rnn_type = 'gru'

        print('gru rnn model')

        rnn_model = RNNEncoder(vocab_size, emb_dim, hid_dim, n_layers, bidirectional, dropout, rnn_type)

        pool_embedder_mean = EmbeddingPooler(hid_dim * 2 if bidirectional else hid_dim, 'mean')
        pool_embedder_max = EmbeddingPooler(hid_dim * 2 if bidirectional else hid_dim, 'max')
        pool_embedder_weighted_mean = EmbeddingPooler(hid_dim * 2 if bidirectional else hid_dim, 'weighted_mean')

        embeddings = rnn_model(tokens)

        print(tokens.shape)
        print(embeddings.shape)
        print(pool_embedder_mean(embeddings).shape)
        print(pool_embedder_max(embeddings).shape)
        print(pool_embedder_weighted_mean(embeddings).shape)

        print('lstm rnn model')

        n_layers = 2
        bidirectional = True
        rnn_type = 'lstm'

        rnn_model = RNNEncoder(vocab_size, emb_dim, hid_dim, n_layers, bidirectional, dropout, rnn_type)

        pool_embedder_mean = EmbeddingPooler(hid_dim * 2 if bidirectional else hid_dim, 'mean')
        pool_embedder_max = EmbeddingPooler(hid_dim * 2 if bidirectional else hid_dim, 'max')
        pool_embedder_weighted_mean = EmbeddingPooler(hid_dim * 2 if bidirectional else hid_dim, 'weighted_mean')

        embeddings = rnn_model(tokens)

        print(tokens.shape)
        print(embeddings.shape)
        print(pool_embedder_mean(embeddings).shape)
        print(pool_embedder_max(embeddings).shape)
        print(pool_embedder_weighted_mean(embeddings).shape)