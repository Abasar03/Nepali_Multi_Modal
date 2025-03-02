import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer
from src.multimodal_embedding_fusion.config import Configuration
from src.multimodal_embedding_fusion.models.model import ProjectionHead,ContrastiveModel
from src.multimodal_embedding_fusion.data.dataset import build_loaders
from src.multimodal_embedding_fusion.utils import make_train_valid_dfs


# class MultiModalFusion(nn.Module): 
#     def __init__(   
#         self,
#         image_embedding=Configuration.image_embedding,
#         text_embedding=Configuration.text_embedding, 
#         fusion_dim=Configuration.fusion_dim
#     ):
#         super().__init__()    
#         self.fusion_dim=fusion_dim  
        
#         self.cross_attention=nn.MultiheadAttention(
#             embed_dim=fusion_dim,
#             num_heads=8,     
#             dropout=0.1
#         )   

#         self.single_modality_proj=nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim*2),
#             nn.LayerNorm(fusion_dim*2),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(fusion_dim*2, fusion_dim*2),
#             nn.LayerNorm(fusion_dim*2)      
#         )           

#         self.final_fusion = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim*2),  
#             nn.LayerNorm(fusion_dim * 2), 
#             nn.LeakyReLU(0.2), 
#             nn.Dropout(0.3),
#             nn.Linear(fusion_dim*2, fusion_dim*2),
#             nn.LayerNorm(fusion_dim*2)
#         )          
               
#     def forward(self,image_projection=None,text_projection=None):     
#         if image_projection is not None and text_projection is not None:
#             if len(image_projection.shape) == 2:
#                 image_projection = image_projection.unsqueeze(0)
#             if len(text_projection.shape) == 2:
#                 text_projection = text_projection.unsqueeze(0)
            
#             fused=self.cross_attention( 
#                 query=image_projection,
#                 key=text_projection,
#                 value=text_projection
#             )[0]
#             return self.final_fusion(fused)
#         else: 
#             fused = self.single_modality_proj(
#             image_projection if image_projection is not None else text_projection
#             )
#             return fused

import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from torch.utils.data import Subset, DataLoader

import torch.nn as nn
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms

class MultiModalFusion(nn.Module):
    def __init__(self,
                 image_embedding=Configuration.image_embedding,
                 text_embedding=Configuration.text_embedding,
                 fusion_dim=Configuration.fusion_dim):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Stronger input transformations
        self.image_input_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        self.text_input_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Multiple attention heads for different aspects
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=0.5
            ) for _ in range(3)  # Using 3 separate attention mechanisms
        ])

        self.attention_weights = nn.Parameter(torch.ones(3) / 3)  # Learnable weights for attention heads

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

        # Separate projections for different semantic spaces
        self.semantic_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(0.5)
            ) for _ in range(2)
        ])

        # Final fusion with residual connections
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),  # Takes concatenated features
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim * 2, fusion_dim * 2)
        )

        # Orthogonal regularization weight
        self.orthogonal_weight = 0.1

    def orthogonal_regularization(self, features):
        # Compute cosine similarity matrix
        features_norm = F.normalize(features, p=2, dim=-1)
        sim_matrix = torch.matmul(features_norm, features_norm.transpose(-2, -1))

        # Remove diagonal elements (self-similarity)
        mask = torch.eye(sim_matrix.size(-1), device=sim_matrix.device)
        sim_matrix = sim_matrix * (1 - mask)

        # Compute loss (want to minimize off-diagonal similarities)
        ortho_loss = torch.mean(torch.abs(sim_matrix))
        return ortho_loss
    
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
    
    

