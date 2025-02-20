import torch
class config:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_tokenizer='NepBERTa/NepBERTa'
    text_encoder='NepBERTa/NepBERTa'
    d_in=1024
    d_out=1024
    qkv_bias=False 
    context_length=128 
    temperature=1.0
    emb_dim = 1024
    num_heads = 8
    num_layers = 6
    vocab_size = 30522
    dropout = 0.1
    max_seq_len=128 
     
 