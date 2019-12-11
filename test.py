import torch

def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, 0).masked_fill(mask == 1, float(1.0))
        return mask

def make_trg_mask(trg_len):
        
        #trg = [batch size, trg len]
        
        #trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(3)
        
        #trg_pad_mask = [batch size, 1, trg len, 1]
                
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        #trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

x = torch.ones(3,3)

print(x)

mask = generate_square_subsequent_mask(3)

print(mask)
print(mask.shape)

print(x * mask)

mask = make_trg_mask(3)

print(mask)
print(mask.shape)

print(x * mask)