import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolEmbedder(nn.Module):
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

        if self.pool_mode == "mean":
            pooled = F.avg_pool1d(embeddings,
                                  kernel_size = seq_len)

        elif self.pool_mode == "max":
            pooled = F.max_pool1d(embeddings,
                                  kernel_size = seq_len)

        elif self.pool_mode == "weighted_mean":
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
            raise ValueError(f"Unknown pool mode: {self.pool_mode}")

        #pooled = [batch size, emb dim, 1]

        pooled = pooled.squeeze(-1)

        #pooled = [batch size, emb dim]

        return pooled

class NeuralBagOfWords(nn.Module):
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

if __name__ == "__main__":
    
    with torch.no_grad():

        vocab_size = 100
        emb_dim = 64
        batch_size = 32
        seq_len = 10

        bag_of_words = NeuralBagOfWords(vocab_size, emb_dim)

        pool_embedder_mean = PoolEmbedder(emb_dim, 'mean')
        pool_embedder_max = PoolEmbedder(emb_dim, 'max')
        pool_embedder_weighted_mean = PoolEmbedder(emb_dim, 'weighted_mean')

        tokens = torch.randint(vocab_size, (seq_len, batch_size))

        embeddings = bag_of_words(tokens)
        
        print(tokens.shape)
        print(embeddings.shape)
        print(pool_embedder_mean(embeddings).shape)
        print(pool_embedder_max(embeddings).shape)
        print(pool_embedder_weighted_mean(embeddings).shape)