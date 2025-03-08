import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.multimodal_text_generation.config import config
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction


def train_model(model,dataloader,valid_loader,num_epochs,device):
  model=model.to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
  criterion=nn.CrossEntropyLoss(ignore_index=1)  

  seq_length = config.max_seq_len
  print(f"Verifying tokenizer settings:")
  print(f"Pad token: {model.tokenizer.pad_token} (ID: {model.tokenizer.pad_token_id})")
  print(f"CLS token: {model.tokenizer.cls_token} (ID: {model.tokenizer.cls_token_id})")
  print(f"SEP token: {model.tokenizer.sep_token} (ID: {model.tokenizer.sep_token_id})")


  best_val_loss = float('inf')
  for epoch in range(num_epochs):  
    model.train()   
    total_loss=0    

    for batch_idx,batch in enumerate(dataloader):
      captions, embeddings = batch
      fused_emb=embeddings.to(device)  
      
      # tokens=model.tokenizer(captions,return_tensors='pt',padding=True,max_length=128,truncation=True)

      tokens=model.tokenizer(captions,return_tensors='pt',padding='max_length',max_length=128,truncation=True,add_special_tokens=True)
      target_ids=tokens['input_ids'].to(device) 

      outputs=model(fused_emb,target_ids[:,:-1])  
      outputs = outputs[:, 1:, :] 
 
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
            
            val_tokens = model.tokenizer(
                val_captions,
                return_tensors='pt',
                padding='max_length',
                max_length=config.max_seq_len,
                truncation=True
            )      
            val_target_ids = val_tokens['input_ids'].to(device)


            val_outputs = model(val_fused_emb, val_target_ids[:, :-1])
            val_outputs = val_outputs[:, 1:, :]  # Match training sequence shift
            loss = criterion(
                val_outputs.reshape(-1, config.vocab_size),
                val_target_ids[:, 1:].contiguous().view(-1)
            )
            val_loss += loss.item()

            
            # generated_ids = model.generate(
            #     val_fused_emb, 
            #     max_length=config.max_seq_len,
            #     num_beams=1,
            #     early_stopping=False
            # )    
            # generated_ids = generated_ids[:, 1:]  # Remove fused embedding position
            # pad_token_id = model.tokenizer.pad_token_id
            # current_length = generated_ids.size(1)
            # if current_length < config.max_seq_len:
            #     padding = torch.full(
            #         (generated_ids.size(0), config.max_seq_len - current_length),
            #         pad_token_id,
            #         device=device
            #     ) 
            #     generated_ids = torch.cat([generated_ids, padding], dim=1)

            # generated_captions = model.tokenizer.batch_decode(
            #     generated_ids, 
            #     skip_special_tokens=True
            # )  
            # print(f'Generated_captions during validation: {generated_captions}')
            
            # all_hypotheses.extend(generated_captions)
            
            # all_references.extend([[ref] for ref in val_captions])

        # bleu_score = corpus_bleu(
        #     all_references,
        #     [h.split() for h in all_hypotheses],
        #     smoothing_function=SmoothingFunction().method1
        # )
        # rouge = Rouge()
        # rouge_scores = rouge.get_scores(
        #     all_hypotheses,
        #     [ref[0] for ref in all_references],  
        #     avg=True
        # ) 
        avg_val_loss = val_loss / len(valid_loader)
        print("\nValidation Metrics:")
        # print(f"BLEU-4 Score: {bleu_score:.4f}")
        
        # print("\nROUGE Scores:")
        # for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
        #     score = rouge_scores[metric]
        #     print(f"{metric.upper():<8} | F1: {score['f']:.4f} | Precision: {score['p']:.4f} | Recall: {score['r']:.4f}")

        print(f"\nAverage Validation Loss: {avg_val_loss:.4f}")
        print("-" * 60)
        
        

