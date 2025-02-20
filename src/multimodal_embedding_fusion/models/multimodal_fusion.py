import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer
from src.multimodal_embedding_fusion.config import Configuration
from src.multimodal_embedding_fusion.models.model import ProjectionHead,ContrastiveModel
from src.multimodal_embedding_fusion.data.dataset import build_loaders
from src.multimodal_embedding_fusion.utils import make_train_valid_dfs


class MultiModalFusion(nn.Module): 
    def __init__(   
        self,
        image_embedding=Configuration.image_embedding,
        text_embedding=Configuration.text_embedding, 
        fusion_dim=Configuration.fusion_dim
    ):
        super().__init__()    
        self.fusion_dim=fusion_dim  
        
        self.cross_attention=nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,     
            dropout=0.1
        )   

        self.single_modality_proj=nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim*2),
            nn.LayerNorm(fusion_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim*2, fusion_dim*2),
            nn.LayerNorm(fusion_dim*2)      
        )           

        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim*2),  
            nn.LayerNorm(fusion_dim * 2), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3),
            nn.Linear(fusion_dim*2, fusion_dim*2),
            nn.LayerNorm(fusion_dim*2)
        )          
               
    def forward(self,image_projection=None,text_projection=None):     
        if image_projection is not None and text_projection is not None:
            if len(image_projection.shape) == 2:
                image_projection = image_projection.unsqueeze(0)
            if len(text_projection.shape) == 2:
                text_projection = text_projection.unsqueeze(0)
            
            fused=self.cross_attention( 
                query=image_projection,
                key=text_projection,
                value=text_projection
            )[0]
            return self.final_fusion(fused)
        else: 
            fused = self.single_modality_proj(
            image_projection if image_projection is not None else text_projection
            )
            return fused


def train_combined(model_path):              
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train") 
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    contrastive_model = ContrastiveModel().to(Configuration.device)
    contrastive_model.load_state_dict(torch.load(model_path))
    contrastive_model.eval()   
    
    fusion_model = MultiModalFusion().to(Configuration.device)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()    
    
    best_loss = float('inf')  
    
    for epoch in range(Configuration.num_epochs): 
        fusion_model.train() 
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
            
            with torch.no_grad():    
                image_features = contrastive_model.image_encoder(batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'] 
                )

            image_projected = contrastive_model.image_projection(image_features)
            text_projected = contrastive_model.text_projection(text_features)
            
            fused = fusion_model(image_projected, text_projected)
            target = torch.cat([image_projected, text_projected], dim=-1) 

            loss = criterion(fused, target)     
            
            optimizer.zero_grad()       
            loss.backward()
            optimizer.step()
             
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')
        
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(fusion_model.state_dict(), 'fused_embeddings.pt')
            
        
        fusion_model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch + 1} - Validation'):
                batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
                
                image_features = contrastive_model.image_encoder(batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                ) 
                
                image_projected = contrastive_model.image_projection(image_features)
                text_projected = contrastive_model.text_projection(text_features)
            
                fused = fusion_model(image_projected, text_projected)
                target = torch.cat([image_projected, text_projected], dim=-1)               
                
                loss = criterion(fused, target) 
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_valid_loss:.4f}')
        
        if avg_valid_loss < best_loss: 
            best_loss = avg_valid_loss 
            torch.save(fusion_model.state_dict(), 'fused_embeddings.pt')
            print(f'Saved best model with Validation Loss: {best_loss:.4f}')

    print('Training completed!') 
    return fusion_model
    
    

