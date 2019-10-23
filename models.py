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
            self.fc = nn.Linear(emb_dim, 1)

    def forward(self, 
                embeddings):

        #embeddings = [batch, emb dim, seq_len]

        _, _, seq_len = embeddings.shape

        if self.pool_mode == 'mean':
            pooled = F.avg_pool1d(embeddings,
                                  kernel_size = seq_len)

        elif self.pool_mode == 'max':
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
                 embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

    def forward(self,
                tokens):

        #tokens = [seq len, batch size]

        embedded = self.embedding(tokens)

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

    def forward(self,
                tokens):

        #tokens = [seq len, batch size]

        embedded = self.embedding(tokens)

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