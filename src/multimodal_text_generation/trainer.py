import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.multimodal_text_generation.config import config
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

def train_model(model,dataloader,valid_loader,num_epochs,device):
  model=model.to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
  criterion=nn.CrossEntropyLoss(ignore_index=1)  

  best_val_loss = float('inf')
  for epoch in range(num_epochs):
    model.train()   
    total_loss=0    

    for batch_idx,batch in enumerate(dataloader):
      captions, embeddings = batch
      fused_emb=embeddings.to(device)  
      
      tokens=model.tokenizer(captions,return_tensors='pt',padding=True,max_length=128,truncation=True)
      target_ids=tokens['input_ids'].to(device) 

      outputs=model(fused_emb,target_ids[:,:-1]) 
      # outputs = outputs[:, 1:, :]
 
      loss = criterion(outputs.reshape(-1, config.vocab_size), target_ids[:, 1:].contiguous().view(-1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss+=loss.item() 

      if batch_idx%1000==0:
        print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss=total_loss/len(dataloader)
    print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}')


    # Validation phase
    model.eval()
    val_loss = 0
    all_hypotheses = []
    all_references = []
    
    with torch.no_grad():
        for val_batch in valid_loader:
            val_captions, val_embeddings = val_batch
            val_fused_emb = val_embeddings.to(device)
            
            # Generate with padding to ensure fixed length
            generated_ids = model.generate(
                val_fused_emb, 
                max_length=128, 
                num_beams=1, 
                early_stopping=False  # Disable early stopping for fixed length
            )
            
            # Pad sequences to match max_length
            pad_token_id = model.tokenizer.pad_token_id
            current_length = generated_ids.size(1)
            pad_token_id = 1
            if current_length < 128:
                padding = torch.full((generated_ids.size(0), 128-current_length), 
                                   pad_token_id, 
                                   device=device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)

            # Rest of validation code remains the same
            generated_captions = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            tokens = model.tokenizer(val_captions, return_tensors='pt', 
                                    padding=True, max_length=128, truncation=True)
            target_ids = tokens['input_ids'].to(device)
            
            # Calculate loss with proper masking
            outputs = model(val_fused_emb, generated_ids[:, :-1])
            loss = criterion(
                outputs.reshape(-1, config.vocab_size), 
                generated_ids[:, 1:].contiguous().view(-1)
            )
            val_loss += loss.item()

            # Store for metrics
            all_hypotheses.extend(generated_captions)
            all_references.extend([[ref.split()] for ref in val_captions])

        bleu_score = corpus_bleu(all_references, [h.split() for h in all_hypotheses])
        rouge = Rouge()
        rouge_scores = rouge.get_scores(all_hypotheses, [ref[0] for ref in all_references], avg=True)

        avg_val_loss = val_loss / len(valid_loader)
        print(f'Val Loss: {avg_val_loss:.4f}') 

